"""The served policy and what every stage-2 rollout request carries (docs/DATA_PLAN.md section 11; src/apex_data/README.md,
"Stage-2 rollout caps and sampling presets").

Endpoint      an OpenAI-compatible server (vLLM on a GPU VM, reached through the SSH tunnel) and the served model id
              (e.g. qwen35-sft); harnesses address it as litellm `openai/<served id>`, the provider prefix both locks pin
              (it keeps reasoning_content in mini-swe-agent's replayed history). `health()` asks /models and can send one
              tiny completion; `require_served()` stops a run whose model is not served.
Sampling      read from the evaluation locks at run time, so rollouts sample exactly as the benchmarks do:
              swe      configs/eval/swebench_pro_v2_hard.yaml inference.sampling (T 0.6, top_p 0.95, top_k 20, min_p 0,
                       presence 0, repetition 1.0) and max_tokens_per_turn
              terminal configs/eval/terminal_bench_2.yaml inference.sampling (T 1.0, top_p 0.95, top_k 20, min_p 0) and
                       max_tokens_per_turn
Caps          STAGE2_CAPS: swe 50 steps / 16,384 tokens per turn / 40,000 output tokens / 20 min; terminal 40 turns / the
              lock's tokens per turn / 60,000 output tokens / 30 min.
Seeds         sample_seed(task_id, k): a fixed 31-bit seed per (task, sample index) derived from paths.SEED, sent with every
              request of that episode, so a resumed or repeated run asks for the same sample and samples of one task differ.
Request log   RequestLogProxy: an in-process HTTP proxy between the harness and the server. Every request body (model,
              seed, sampling, messages; api keys removed) and the assembled response (content, reasoning, tool calls,
              finish_reason, usage) go to one JSONL(.gz) file per episode. It also enforces the episode's output-token cap:
              a request's max_tokens is clamped to the remaining budget (the original value is logged) and, once the budget
              is spent, the proxy answers 429 with BUDGET_MARKER, which the drivers map to exit_reason token_cap.

Only the standard library (plus PyYAML for the locks) is used, so this module imports in the train and harbor venvs.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from apex_data import paths

LITELLM_PREFIX = "openai/"
DEFAULT_API_KEY_ENV = "EVAL_API_KEY"   # name of the env var holding the endpoint key (a dummy for vLLM); never a flag value
BUDGET_MARKER = "APEX_OUTPUT_TOKEN_BUDGET_EXCEEDED"
SECRET_KEYS = frozenset({"api_key", "apikey", "authorization", "x-api-key", "openai_api_key"})
HOP_HEADERS = frozenset({"host", "content-length", "connection", "transfer-encoding", "accept-encoding", "keep-alive"})
FAMILIES = ("swe", "terminal")
SCAFFOLD_FOR_FAMILY = {"swe": "mini_swe_toolcall", "terminal": "terminus2"}
FAMILY_OF_SCAFFOLD = {v: k for k, v in SCAFFOLD_FOR_FAMILY.items()}
LOCKS = {"swe": paths.CONFIGS / "eval" / "swebench_pro_v2_hard.yaml", "terminal": paths.CONFIGS / "eval" / "terminal_bench_2.yaml"}


@dataclass(frozen=True)
class Caps:
    """Per-episode caps of the stage-2 rollouts (plan section 11)."""

    max_steps: int                     # model calls (mini-swe-agent step_limit / Terminus-2 max_turns)
    max_tokens_per_turn: int | None    # max_tokens per call; None = the family's lock value
    max_output_tokens: int             # completion tokens per episode (enforced by the proxy)
    wall_sec: int                      # agent wall clock (mini-swe-agent wall_time_limit_seconds / Harbor agent timeout)


STAGE2_CAPS: dict[str, Caps] = {
    "swe": Caps(max_steps=50, max_tokens_per_turn=16_384, max_output_tokens=40_000, wall_sec=20 * 60),
    "terminal": Caps(max_steps=40, max_tokens_per_turn=None, max_output_tokens=60_000, wall_sec=30 * 60),
}


def lock_inference(family: str) -> dict[str, Any]:
    """{"sampling": {...}, "max_tokens_per_turn": int, "lock": path} from the family's evaluation lock (read-only)."""
    import yaml  # noqa: PLC0415 - both venvs have it; keeps the module importable for pure helpers

    lock = yaml.safe_load(LOCKS[family].read_text())
    inf = lock["inference"]
    return {"sampling": dict(inf["sampling"]), "max_tokens_per_turn": int(inf["max_tokens_per_turn"]),
            "lock": str(LOCKS[family].relative_to(paths.REPO))}


def caps_for(family: str) -> Caps:
    """STAGE2_CAPS[family] with a None per-turn limit resolved to the lock's max_tokens_per_turn."""
    caps = STAGE2_CAPS[family]
    if caps.max_tokens_per_turn is None:
        caps = dataclasses.replace(caps, max_tokens_per_turn=lock_inference(family)["max_tokens_per_turn"])
    return caps


def sample_seed(task_id: str, sample_index: int, base: int = paths.SEED) -> int:
    """Request seed of sample `sample_index` of `task_id`: first 31 bits of sha256('<base>:<task_id>:<k>')."""
    return int(hashlib.sha256(f"{base}:{task_id}:{int(sample_index)}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def episode_id(task_id: str, round_: str, policy: str, sample_index: int) -> str:
    """schema contract: '<task_id>#<round>#<policy>#<sample_index>'."""
    return f"{task_id}#{round_}#{policy}#{int(sample_index)}"


@dataclass(frozen=True)
class Endpoint:
    """Where the policy is served and how the harnesses address it."""

    api_base: str                       # e.g. http://172.18.0.1:8000/v1
    model_name: str                     # served model id, e.g. qwen35-sft
    api_key_env: str = DEFAULT_API_KEY_ENV

    @property
    def litellm_model(self) -> str:
        return LITELLM_PREFIX + self.model_name

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env) or "EMPTY"

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Endpoint":
        return cls(api_base=d["api_base"], model_name=d["model_name"], api_key_env=d.get("api_key_env", DEFAULT_API_KEY_ENV))

    def _request(self, path: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        url = self.api_base.rstrip("/") + path
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key()}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - operator-given endpoint
            return json.loads(r.read().decode())

    def health(self, *, probe: bool = False, timeout: float = 30.0) -> dict[str, Any]:
        """{"ok", "models", "served", "probe", "error"}: /models answers and lists model_name; with probe, one completion
        (max_tokens 1, the lock's T 0.6 preset, seed 0) returns 200."""
        out: dict[str, Any] = {"api_base": self.api_base, "model_name": self.model_name, "ok": False, "models": None, "served": False}
        try:
            out["models"] = [m.get("id") for m in self._request("/models", None, timeout).get("data", [])]
            out["served"] = self.model_name in out["models"]
        except (urllib.error.URLError, OSError, ValueError) as e:
            out["error"] = f"/models: {type(e).__name__}: {e}"
            return out
        if probe and out["served"]:
            body = {"model": self.model_name, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 1,
                    "temperature": 0.6, "seed": 0}
            t0 = time.time()
            try:
                resp = self._request("/chat/completions", body, timeout)
                out["probe"] = {"latency_ms": round((time.time() - t0) * 1000), "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason")}
            except (urllib.error.URLError, OSError, ValueError) as e:
                out["error"] = f"probe: {type(e).__name__}: {e}"
                return out
        out["ok"] = out["served"]
        return out

    def require_served(self, *, probe: bool = False) -> dict[str, Any]:
        h = self.health(probe=probe)
        if not h["ok"]:
            raise RuntimeError(f"endpoint {self.api_base} does not serve {self.model_name!r}: {h.get('error') or h.get('models')}")
        return h


def redact(obj: Any) -> Any:
    """Copy of a JSON-like object without api-key-like fields (keys compared case-insensitively)."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items() if str(k).lower() not in SECRET_KEYS}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


# --- token accounting and response normalisation -------------------------------------------------------------------------
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_requests: int = 0                # answered upstream with 200
    n_length_finish: int = 0           # responses with finish_reason == "length"
    n_errors: int = 0                  # upstream errors and budget refusals returned to the harness

    def add(self, usage: dict[str, Any] | None, finish_reason: str | None) -> None:
        self.n_requests += 1
        self.n_length_finish += int(finish_reason == "length")
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def reasoning_of(msg: dict[str, Any]) -> str | None:
    """Reasoning text of an OpenAI-style message: reasoning_content, else vLLM's `reasoning`, else litellm's
    provider_specific_fields copy."""
    for key in ("reasoning_content", "reasoning"):
        if isinstance(msg.get(key), str):
            return msg[key]
    psf = msg.get("provider_specific_fields") or {}
    for key in ("reasoning_content", "reasoning"):
        if isinstance(psf.get(key), str):
            return psf[key]
    return None


def normalise_message(msg: dict[str, Any]) -> dict[str, Any]:
    """OpenAI message -> {role, content, reasoning_content, tool_calls}."""
    calls = [{"id": tc.get("id"), "type": tc.get("type") or "function",
              "function": {"name": (tc.get("function") or {}).get("name"), "arguments": (tc.get("function") or {}).get("arguments")}}
             for tc in msg.get("tool_calls") or []]
    return {"role": msg.get("role") or "assistant", "content": msg.get("content"), "reasoning_content": reasoning_of(msg),
            "tool_calls": calls or None}


def message_from_completion(resp: dict[str, Any]) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    return {"message": normalise_message(choice.get("message") or {}), "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage")}


def assemble_stream(sse_text: str) -> dict[str, Any]:
    """A streamed chat.completion (text/event-stream) reassembled into {message, finish_reason, usage}."""
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish, usage = None, None
    for line in sse_text.splitlines():
        if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"):
            continue
        try:
            chunk = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            piece = delta.get("reasoning_content") if delta.get("reasoning_content") is not None else delta.get("reasoning")
            if piece:
                reasoning.append(piece)
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(int(tc.get("index", 0)), {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["function"]["name"] = fn.get("name") or slot["function"]["name"]
                slot["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    message = {"role": "assistant", "content": "".join(content) if content else None,
               "reasoning_content": "".join(reasoning) if reasoning else None, "tool_calls": [calls[i] for i in sorted(calls)] or None}
    return {"message": message, "finish_reason": finish, "usage": usage}


# --- the logging proxy -----------------------------------------------------------------------------------------------------
PROXY_PREFIX = "/v1"   # the proxy's api_base path; requests under it go to the upstream api_base (which carries its own /v1)

class _ProxyState:
    def __init__(self, upstream: str, log_path: Path, output_token_budget: int | None, upstream_timeout: float):
        self.upstream = upstream.rstrip("/")
        self.log_path = log_path
        self.output_token_budget = output_token_budget
        self.upstream_timeout = upstream_timeout
        self.usage = TokenUsage()
        self.lock = threading.Lock()
        self.budget_hits = 0
        self.clamped = 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = gzip.open(log_path, "at", encoding="utf-8") if log_path.suffix == ".gz" else log_path.open("a", encoding="utf-8")
        self._i = 0

    def record(self, rec: dict[str, Any]) -> None:
        with self.lock:
            self._i += 1
            self._log.write(json.dumps({"i": self._i, **rec}, ensure_ascii=False) + "\n")
            self._log.flush()

    def close(self) -> None:
        with self.lock:
            self._log.close()


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _ProxyState):
        super().__init__(address, _ProxyHandler)
        self.state = state


class _ProxyHandler(BaseHTTPRequestHandler):
    server: _ProxyServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:   # the JSONL record replaces the access log
        pass

    def _upstream_headers(self) -> dict[str, str]:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Accept-Encoding"] = "identity"
        return headers

    def _upstream_url(self) -> str:
        """The upstream URL of this request. The proxy serves the API under `/v1` (its `url`), the upstream api_base
        already ends in its own version prefix, so the proxy's `/v1` is replaced by the upstream api_base:
        <proxy>/v1/chat/completions -> <upstream api_base>/chat/completions."""
        path = self.path
        if path == PROXY_PREFIX or path.startswith(PROXY_PREFIX + "/") or path.startswith(PROXY_PREFIX + "?"):
            path = path[len(PROXY_PREFIX):]
        return self.server.state.upstream + path

    def _send(self, status: int, payload: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        req = urllib.request.Request(self._upstream_url(), headers=self._upstream_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
                self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read(), e.headers.get("Content-Type", "application/json"))
        except (urllib.error.URLError, OSError) as e:
            self._send(502, json.dumps({"error": {"message": f"upstream unreachable: {e}", "type": "proxy_error"}}).encode())

    def _budget(self, body: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """(body to forward, refusal message or None): refuse once the budget is spent, else clamp max_tokens to it."""
        st = self.server.state
        if st.output_token_budget is None:
            return body, None
        with st.lock:
            remaining = st.output_token_budget - st.usage.completion_tokens
            if remaining <= 0:
                st.budget_hits += 1
                return body, f"{BUDGET_MARKER}: {st.usage.completion_tokens} completion tokens >= budget {st.output_token_budget}"
            clamped = dict(body)
            for key in ("max_tokens", "max_completion_tokens"):
                if isinstance(body.get(key), int) and body[key] > remaining:
                    clamped[key] = remaining
            if clamped != body:
                st.clamped += 1
            return clamped, None

    def do_POST(self) -> None:
        st = self.server.state
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {"_raw": raw.decode(errors="replace")}
        is_dict = isinstance(body, dict)
        stream = bool(body.get("stream")) if is_dict else False
        t0 = time.time()
        fwd, refusal = self._budget(body) if is_dict else (body, None)
        rec: dict[str, Any] = {"t": round(t0, 3), "path": self.path, "stream": stream, "has_tools": bool(is_dict and body.get("tools")),
                               "body": redact(fwd)}
        if fwd is not body and fwd != body:
            rec["clamped_from"] = {k: body[k] for k in ("max_tokens", "max_completion_tokens") if k in body and fwd.get(k) != body[k]}
            raw = json.dumps(fwd).encode()
        if refusal is not None:
            with st.lock:
                st.usage.n_errors += 1
            rec.update(status=429, latency_ms=0, response={"error": refusal})
            st.record(rec)
            self._send(429, json.dumps({"error": {"message": refusal, "type": "budget_exceeded", "code": "apex_budget"}}).encode())
            return
        req = urllib.request.Request(self._upstream_url(), data=raw, headers=self._upstream_headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=st.upstream_timeout) as r:  # noqa: S310
                ctype = r.headers.get("Content-Type", "application/json")
                if stream and "text/event-stream" in ctype:
                    parsed = assemble_stream(self._relay_stream(r, ctype))
                else:
                    payload = r.read()
                    self._send(r.status, payload, ctype)
                    try:
                        parsed = message_from_completion(json.loads(payload))
                    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                        parsed = {"error": "unparseable response", "raw": payload[:2000].decode(errors="replace")}
                status = r.status
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
            self._send(status, payload, e.headers.get("Content-Type", "application/json"))
            parsed = {"error": payload[:2000].decode(errors="replace")}
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            status, parsed = 502, {"error": f"upstream unreachable: {e}"}
            self._send(status, json.dumps({"error": {"message": parsed["error"], "type": "proxy_error"}}).encode())
        with st.lock:
            if status == 200 and "error" not in parsed:
                st.usage.add(parsed.get("usage"), parsed.get("finish_reason"))
            else:
                st.usage.n_errors += 1
        rec.update(status=status, latency_ms=round((time.time() - t0) * 1000), response=parsed)
        st.record(rec)

    def _relay_stream(self, r: Any, ctype: str) -> str:
        """Forward an event stream chunk by chunk (chunked encoding) and return the whole text."""
        self.send_response(r.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        parts: list[bytes] = []
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            parts.append(chunk)
            self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        return b"".join(parts).decode("utf-8", errors="replace")


class RequestLogProxy:
    """Logging, budget-enforcing proxy in front of an OpenAI-compatible server. Context manager; `url` is the api_base to
    give the harness (bound to 127.0.0.1: the harnesses' LLM loops run on the host)."""

    def __init__(self, upstream_api_base: str, log_path: Path | str, *, output_token_budget: int | None = None,
                 upstream_timeout: float = 1800.0, bind_host: str = "127.0.0.1"):
        self.state = _ProxyState(upstream_api_base, Path(log_path), output_token_budget, upstream_timeout)
        self.bind_host = bind_host
        self._server: _ProxyServer | None = None
        self.port: int | None = None

    def __enter__(self) -> "RequestLogProxy":
        self._server = _ProxyServer((self.bind_host, 0), self.state)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self.state.close()

    @property
    def url(self) -> str:
        return f"http://{self.bind_host}:{self.port}{PROXY_PREFIX}"

    @property
    def usage(self) -> TokenUsage:
        return self.state.usage

    def summary(self) -> dict[str, Any]:
        return {"usage": self.usage.as_dict(), "budget_hits": self.state.budget_hits, "clamped_requests": self.state.clamped,
                "requests_log": str(self.state.log_path)}


def read_request_log(path: Path | str) -> Iterator[dict[str, Any]]:
    """Records of a proxy log (.jsonl or .jsonl.gz); a log cut short by a crash yields its complete records only."""
    path = Path(path)
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        return
    except EOFError:
        return


def answered(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The 200-answered model calls of a proxy log, in order: {i, message, finish_reason, usage, latency_ms, seed}."""
    out = []
    for r in records:
        resp = r.get("response") or {}
        if r.get("status") == 200 and isinstance(resp.get("message"), dict):
            out.append({"i": r.get("i"), "message": resp["message"], "finish_reason": resp.get("finish_reason"),
                        "usage": resp.get("usage"), "latency_ms": r.get("latency_ms"), "seed": (r.get("body") or {}).get("seed")})
    return out
