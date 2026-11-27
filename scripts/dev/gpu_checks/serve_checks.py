#!/usr/bin/env python
"""Serving-VM checks against a running vLLM endpoint (helpers of scripts/serve_vllm.sh): smoke, S3, S5, S6, S7.

Run from the repository root (normally via scripts/serve_vllm.sh <subcommand>):
    .venv/train/bin/python scripts/dev/gpu_checks/serve_checks.py smoke --base-url http://127.0.0.1:8000 --model qwen35-base
    .venv/train/bin/python scripts/dev/gpu_checks/serve_checks.py s3 --base-url ... --model ... --tokenizer <ckpt dir>
        [--prompt-tokens 250000 --max-tokens 64] [--container vllm-<name>-<port>]
    .venv/train/bin/python scripts/dev/gpu_checks/serve_checks.py s5 --base-url ... --model ... [--n-tool 20 --n-plain 20]
    .venv/train/bin/python scripts/dev/gpu_checks/serve_checks.py s6 --base-url ... --model ... --tokenizer <ckpt dir> [--n 10]
    .venv/train/bin/python scripts/dev/gpu_checks/serve_checks.py s7 --base-url ... --model ... --hf-reference <T6 jsonl>
Inputs : the endpoint (OpenAI-compatible, vLLM v0.31.0 flags of serve_vllm.sh), the checkpoint's tokenizer for
         S3/S6, the own prompt set (prompts_own.jsonl) for S5, T6's hf_greedy_reference.jsonl for S7. No benchmark data.
Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl (one row per request), errors.log, resources.csv}
         plus raw responses (smoke_*.json). Exit 0 PASS, 1 FAIL.
Pass criteria (REPORT.md 3c):
  smoke  one thinking request -> non-empty `message.reasoning` (vLLM 0.31 field name) and content without <think>
         leaks; one tool request -> `tool_calls` with a known function name and JSON-decodable arguments.
  S3     one ~250K-token synthetic prompt, max_tokens 64: HTTP 200, prompt_tokens >= 95 % of the target, TTFT
         logged; no preemption / OOM lines in the container log (when --container is given).
  S5     20 tool + 20 plain own prompts, streaming and not: >= 95 % of tool prompts yield parsable tool_calls
         (valid JSON arguments), no <think>/<tool_call> leaks in content, expected tool name matched >= 90 %.
  S6     the 9 synthetic conversations: HF apply_chat_template ids == server /tokenize ids with the same messages,
         tools and chat_template_kwargs (generation prompt and full render).
  S7     HF greedy reference vs server (temperature 0 with neutral penalties and truncation per request, so the
         serving lock's --override-generation-config defaults do not turn "greedy" into penalized sampling;
         return_token_ids): >= 68 % of prompts identical over the first 64 tokens and >= 76 % mean per-token
         agreement. Calibrated on the base model on a B200 (vLLM v0.31.0, the lock's flags, 2026-10-12): 39/50
         identical, 0.81 agreement; bf16 kernel differences between HF and vLLM make long greedy runs diverge, so
         the bar is the base model's level minus 10 and 5 points, not near-identity.
GPU-only: all of them need the served model; the request/parse plumbing was exercised against a fake server on CPU.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from loading import load_tokenizer, read_prompts, render_ids, synthetic_conversations
from runlog import CheckRun

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts_own.jsonl"
LEAK_RE = re.compile(r"<think>|</think>|<tool_call>|</tool_call>")
BAD_LOG_RE = re.compile(r"preempt|out of memory|OutOfMemory|CUDA error", re.IGNORECASE)
THINK_KW = {"enable_thinking": True}


# ----------------------------------------------------------------------------------------------- HTTP helpers
def post_json(url: str, body: dict, timeout: float = 600.0) -> tuple[int, dict | None, float]:
    """POST JSON; returns (status, parsed body or {"error": text}, seconds)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return r.status, json.loads(r.read().decode()), time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")[:2000]}, time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - t0


def stream_sse(url: str, body: dict, timeout: float = 600.0) -> Iterator[tuple[dict, float]]:
    """POST with stream=true; yields (chunk json, seconds since request start) per SSE data line."""
    data = json.dumps({**body, "stream": True}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            yield json.loads(payload), time.perf_counter() - t0


def chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/chat/completions"


def message_of(resp: dict) -> dict:
    return resp["choices"][0]["message"]


def reasoning_of(msg: dict) -> str:
    return msg.get("reasoning") or msg.get("reasoning_content") or ""


def parse_tool_calls(tool_calls: list[dict] | None) -> tuple[list[str], bool]:
    """(function names, all arguments JSON-decodable)."""
    names, ok = [], True
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        names.append(fn.get("name"))
        try:
            args = fn.get("arguments")
            json.loads(args) if isinstance(args, str) else dict(args or {})
        except Exception:  # noqa: BLE001
            ok = False
    return names, ok


def collect_stream(url: str, body: dict) -> dict[str, Any]:
    """Accumulate a streamed chat completion into {content, reasoning, tool_calls, ttft_s, total_s, n_chunks}."""
    content, reasoning, calls, ttft, total, n = [], [], {}, None, 0.0, 0
    for chunk, t in stream_sse(url, body):
        n += 1
        total = t
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning") or delta.get("tool_calls")):
                ttft = t
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning") or delta.get("reasoning_content"):
                reasoning.append(delta.get("reasoning") or delta.get("reasoning_content"))
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": None, "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] = fn.get("name") or slot["name"]
                slot["arguments"] += fn.get("arguments") or ""
    tool_calls = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for _, c in sorted(calls.items())]
    return {"content": "".join(content), "reasoning": "".join(reasoning), "tool_calls": tool_calls, "ttft_s": ttft,
            "total_s": total, "n_chunks": n}


def container_log_problems(container: str | None, since: str) -> list[str] | None:
    """Preemption / OOM lines of the container since `since`; None when the container log is unavailable."""
    if not container:
        return None
    out = subprocess.run(["docker", "logs", "--since", since, container], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return None
    return [ln for ln in (out.stdout + out.stderr).splitlines() if BAD_LOG_RE.search(ln)][:20]


# ------------------------------------------------------------------------------------------------- subcommands
def cmd_smoke(args: argparse.Namespace, run: CheckRun) -> bool:
    url = chat_url(args.base_url)
    plain = read_prompts(Path(args.prompts), kind="plain", limit=1)[0]
    tool = read_prompts(Path(args.prompts), kind="tool", limit=1)[0]
    body_think = {"model": args.model, "messages": plain["messages"], "max_tokens": args.max_tokens, "temperature": 0.0,
                  "seed": 0, "chat_template_kwargs": THINK_KW}
    body_tool = {"model": args.model, "messages": tool["messages"], "tools": tool["tools"], "max_tokens": args.max_tokens,
                 "temperature": 0.0, "seed": 0, "chat_template_kwargs": THINK_KW,
                 "tool_choice": "auto" if args.tool_choice == "auto" else {"type": "function", "function": {"name": tool["expect_tool"]}}}
    results = {}
    for name, body in (("thinking", body_think), ("tool_call", body_tool)):
        status, resp, secs = post_json(url, body)
        (run.run_dir / f"smoke_{name}.json").write_text(json.dumps(resp, indent=2))
        if status != 200:
            run.error(f"smoke {name}: HTTP {status} {resp}")
            results[name] = {"ok": False}
            continue
        msg = message_of(resp)
        names, args_ok = parse_tool_calls(msg.get("tool_calls"))
        rec = {"event": f"smoke_{name}", "status": status, "seconds": round(secs, 2), "reasoning_chars": len(reasoning_of(msg)),
               "content_chars": len(msg.get("content") or ""), "tool_calls": names, "arguments_json_ok": args_ok,
               "leak": bool(LEAK_RE.search(msg.get("content") or "")), "finish_reason": resp["choices"][0].get("finish_reason"),
               "usage": resp.get("usage")}
        run.metric(rec)
        run.log(f"smoke {name}: {rec}")
        if name == "thinking":
            rec["ok"] = rec["reasoning_chars"] > 0 and not rec["leak"] and rec["content_chars"] > 0
        else:
            rec["ok"] = bool(names) and args_ok and names[0] in {t["function"]["name"] for t in tool["tools"]} and not rec["leak"]
        if not rec["ok"]:
            run.error(f"smoke {name} failed: {rec}")
        results[name] = rec
    return all(r.get("ok") for r in results.values()) and len(results) == 2


def build_long_prompt(tok, target_tokens: int) -> str:
    """Own filler text repeated until the chat-rendered prompt holds ~target_tokens tokens."""
    unit = ("Entry {i}: the pipeline stage reported status ok, latency 12 ms, queue depth 3, retries 0; "
            "operator note: nothing to do, continue monitoring the next window. ")
    sample = tok(unit.format(i=12345), add_special_tokens=False)["input_ids"]
    n = max(1, target_tokens // len(sample))
    text = "".join(unit.format(i=i) for i in range(n))
    return text + "\n\nIn one short sentence: what was the status of entry 7?"


def cmd_s3(args: argparse.Namespace, run: CheckRun) -> bool:
    tok = load_tokenizer(args.tokenizer)
    prompt = build_long_prompt(tok, args.prompt_tokens)
    messages = [{"role": "user", "content": prompt}]
    n_est = len(render_ids(tok, messages, None, add_generation_prompt=True, **THINK_KW))
    run.log(f"synthetic prompt: ~{n_est:,d} tokens ({len(prompt):,d} chars)")
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens, "temperature": 0.0, "seed": 0,
            "chat_template_kwargs": THINK_KW}
    try:
        s = collect_stream(chat_url(args.base_url), body)
        ok_http = True
    except urllib.error.HTTPError as exc:
        run.error(f"S3 HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}")
        s, ok_http = {"ttft_s": None, "total_s": None, "n_chunks": 0}, False
    problems = container_log_problems(args.container, since)
    if problems is None:
        run.log("container log not available: preemption/OOM lines not checked (pass --container on the serving VM)")
    rec = {"event": "s3", "prompt_tokens_est": n_est, "ttft_s": s.get("ttft_s"), "total_s": s.get("total_s"),
           "n_chunks": s.get("n_chunks"), "log_problems": problems}
    run.metric(rec)
    run.log(f"S3: {rec}")
    status, resp, _ = post_json(chat_url(args.base_url), {**body, "max_tokens": 1}) if ok_http else (0, {}, 0)
    usage = (resp or {}).get("usage") or {}
    run.metric({"event": "s3_usage", "status": status, "usage": usage})
    ok = ok_http and not problems and usage.get("prompt_tokens", 0) >= 0.95 * n_est
    if not ok:
        run.error(f"S3 failed: http={ok_http} problems={problems} usage={usage}")
    return ok


def one_s5(url: str, model: str, p: dict, max_tokens: int, streaming: bool) -> dict[str, Any]:
    body = {"model": model, "messages": p["messages"], "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
            "chat_template_kwargs": THINK_KW}
    if p["kind"] == "tool":
        body.update(tools=p["tools"], tool_choice="auto")
    try:
        if streaming:
            s = collect_stream(url, body)
            msg = {"content": s["content"], "reasoning": s["reasoning"], "tool_calls": s["tool_calls"]}
            status, secs = 200, s["total_s"]
        else:
            status, resp, secs = post_json(url, body)
            msg = message_of(resp) if status == 200 else {}
    except Exception as exc:  # noqa: BLE001
        status, msg, secs = 0, {"error": f"{type(exc).__name__}: {exc}"}, 0.0
    names, args_ok = parse_tool_calls(msg.get("tool_calls"))
    content = msg.get("content") or ""
    return {"event": "s5", "id": p["id"], "kind": p["kind"], "streaming": streaming, "status": status, "seconds": round(secs or 0, 2),
            "tool_calls": names, "arguments_json_ok": args_ok, "expected_tool": p.get("expect_tool"),
            "name_match": bool(names) and names[0] == p.get("expect_tool"), "leak": bool(LEAK_RE.search(content)),
            "reasoning_chars": len(reasoning_of(msg)), "content_chars": len(content), "error": msg.get("error")}


def cmd_s5(args: argparse.Namespace, run: CheckRun) -> bool:
    prompts = read_prompts(Path(args.prompts), kind="tool", limit=args.n_tool) + read_prompts(Path(args.prompts), kind="plain", limit=args.n_plain)
    jobs = [(p, streaming) for p in prompts for streaming in (False, True)]
    url = chat_url(args.base_url)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(lambda j: one_s5(url, args.model, j[0], args.max_tokens, j[1]), jobs))
    for r in rows:
        run.metric(r)
    tool_rows = [r for r in rows if r["kind"] == "tool"]
    plain_rows = [r for r in rows if r["kind"] == "plain"]
    parsable = sum(1 for r in tool_rows if r["status"] == 200 and r["tool_calls"] and r["arguments_json_ok"])
    matched = sum(1 for r in tool_rows if r["name_match"])
    leaks = sum(1 for r in rows if r["leak"])
    http_ok = sum(1 for r in rows if r["status"] == 200)
    plain_ok = sum(1 for r in plain_rows if r["status"] == 200 and r["content_chars"] > 0)
    summary = {"n_tool_requests": len(tool_rows), "parsable_tool_calls": parsable, "name_matches": matched, "leaks": leaks,
               "http_ok": http_ok, "n_requests": len(rows), "plain_with_content": plain_ok, "n_plain_requests": len(plain_rows)}
    run.log(f"S5: {summary}")
    run.metric({"event": "s5_summary", **summary})
    ok = (http_ok == len(rows) and leaks == 0 and parsable >= 0.95 * len(tool_rows) and matched >= args.min_name_match * len(tool_rows)
          and plain_ok == len(plain_rows))
    if not ok:
        run.error(f"S5 failed: {summary}; first errors: {[r['error'] for r in rows if r['error']][:3]}")
    return ok


def cmd_s6(args: argparse.Namespace, run: CheckRun) -> bool:
    tok = load_tokenizer(args.tokenizer)
    url = args.base_url.rstrip("/") + "/tokenize"
    convs = synthetic_conversations()[:args.n]
    n_ok = 0
    for conv in convs:
        for gen_prompt, msgs in ((True, conv["messages"][:-1]), (False, conv["messages"])):
            kw = {"enable_thinking": True, **conv["kwargs"]}
            hf = render_ids(tok, msgs, conv["tools"], add_generation_prompt=gen_prompt, **kw)
            body = {"model": args.model, "messages": msgs, "add_generation_prompt": gen_prompt, "chat_template_kwargs": kw}
            if conv["tools"]:
                body["tools"] = conv["tools"]
            status, resp, _ = post_json(url, body, timeout=120)
            srv = resp.get("tokens") if status == 200 else None
            same = srv == hf
            first_diff = next((i for i, (a, b) in enumerate(zip(hf, srv or [])) if a != b), min(len(hf), len(srv or [])) if srv is not None else None)
            rec = {"event": "s6", "conversation": conv["name"], "generation_prompt": gen_prompt, "status": status, "n_hf": len(hf),
                   "n_server": len(srv) if srv is not None else None, "identical": same, "first_diff_index": None if same else first_diff}
            if not same and srv is not None and first_diff is not None:
                rec["hf_context"] = tok.decode(hf[max(0, first_diff - 12): first_diff + 12])
                rec["server_context"] = tok.decode(srv[max(0, first_diff - 12): first_diff + 12])
            if status != 200:
                rec["error"] = resp.get("error")
            run.metric(rec)
            run.log(f"S6 {conv['name']} gen_prompt={gen_prompt}: identical={same} (hf {len(hf)} / server {rec['n_server']})")
            n_ok += same
    total = 2 * len(convs)
    ok = n_ok == total
    if not ok:
        run.error(f"S6: {total - n_ok}/{total} renders differ between HF apply_chat_template and /tokenize")
    run.metric({"event": "s6_summary", "identical": n_ok, "total": total})
    return ok


# Per-request sampling that makes temperature 0 a true greedy decode on a server whose lock sets
# --override-generation-config (presence_penalty 1.5, top_k 20, top_p 0.95): request values win over those defaults.
GREEDY_NEUTRAL = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0, "top_k": -1, "top_p": 1.0,
                  "min_p": 0.0}


def cmd_s7(args: argparse.Namespace, run: CheckRun) -> bool:
    rows = [json.loads(ln) for ln in Path(args.hf_reference).read_text().splitlines() if ln.strip()]
    url = chat_url(args.base_url)

    def one(row: dict) -> dict[str, Any]:
        body = {"model": args.model, "messages": row["messages"], "max_tokens": args.max_tokens, "temperature": 0.0, "seed": 0,
                "chat_template_kwargs": row.get("chat_template_kwargs", THINK_KW), "return_token_ids": True, **GREEDY_NEUTRAL}
        status, resp, secs = post_json(url, body)
        if status != 200:
            return {"event": "s7", "id": row["id"], "status": status, "error": resp.get("error")}
        choice = resp["choices"][0]
        srv = choice.get("token_ids")
        if srv is None:
            return {"event": "s7", "id": row["id"], "status": status, "error": "response has no token_ids (return_token_ids unsupported?)"}
        ref = row["output_token_ids"]
        k = min(args.first_k, len(ref), len(srv))
        n = min(len(ref), len(srv))
        agree = sum(1 for a, b in zip(ref[:n], srv[:n]) if a == b)
        return {"event": "s7", "id": row["id"], "status": status, "seconds": round(secs, 2), "n_ref": len(ref), "n_server": len(srv),
                "identical_first_k": ref[:k] == srv[:k], "k": k, "per_token_agreement": agree / n if n else None,
                "prompt_tokens_server": (resp.get("usage") or {}).get("prompt_tokens"), "prompt_tokens_hf": len(row["prompt_token_ids"]),
                "finish_reason": choice.get("finish_reason")}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(one, rows))
    for r in results:
        run.metric(r)
    good = [r for r in results if r.get("status") == 200 and "error" not in r]
    ident = sum(1 for r in good if r["identical_first_k"])
    agreements = [r["per_token_agreement"] for r in good if r["per_token_agreement"] is not None]
    prompt_mismatch = sum(1 for r in good if r["prompt_tokens_server"] not in (None, r["prompt_tokens_hf"]))
    summary = {"n": len(rows), "n_ok_http": len(good), "identical_first_k": ident, "frac_identical": round(ident / len(rows), 3) if rows else None,
               "mean_token_agreement": round(sum(agreements) / len(agreements), 4) if agreements else None, "prompt_token_count_mismatch": prompt_mismatch}
    run.log(f"S7: {summary}")
    run.metric({"event": "s7_summary", **summary})
    ok = bool(rows) and len(good) == len(rows) and ident >= args.min_identical * len(rows) and (summary["mean_token_agreement"] or 0) >= args.min_token_agreement
    if not ok:
        run.error(f"S7 failed: {summary}; errors: {[r.get('error') for r in results if r.get('error')][:3]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", default="http://127.0.0.1:8000")
        p.add_argument("--model", required=True, help="served model name")
        p.add_argument("--run-name", default=None)
        p.add_argument("--serve-run-dir", default=None, help="logs/<run_id> of the serve_vllm.sh start run, for cross-reference")

    p = sub.add_parser("smoke"); common(p)
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS)); p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--tool-choice", choices=["auto", "forced"], default="auto")
    p = sub.add_parser("s3"); common(p)
    p.add_argument("--tokenizer", required=True); p.add_argument("--prompt-tokens", type=int, default=250_000)
    p.add_argument("--max-tokens", type=int, default=64); p.add_argument("--container", default=None)
    p = sub.add_parser("s5"); common(p)
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS)); p.add_argument("--n-tool", type=int, default=20)
    p.add_argument("--n-plain", type=int, default=20); p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=8); p.add_argument("--min-name-match", type=float, default=0.9)
    p = sub.add_parser("s6"); common(p)
    p.add_argument("--tokenizer", required=True); p.add_argument("--n", type=int, default=10)
    p = sub.add_parser("s7"); common(p)
    p.add_argument("--hf-reference", required=True); p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--first-k", type=int, default=64); p.add_argument("--min-identical", type=float, default=0.68)
    p.add_argument("--min-token-agreement", type=float, default=0.76); p.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    run = CheckRun(args.run_name or f"serve_{args.cmd}", config=vars(args))
    run.write_metadata(serve_run_dir=args.serve_run_dir, endpoint=args.base_url, served_model=args.model)
    ok = {"smoke": cmd_smoke, "s3": cmd_s3, "s5": cmd_s5, "s6": cmd_s6, "s7": cmd_s7}[args.cmd](args, run)
    return run.finish("PASS" if ok else "FAIL", {"endpoint": args.base_url, "model": args.model}, gpu_count=0)


if __name__ == "__main__":
    sys.exit(main())
