"""Local OpenAI-compatible forwarding proxy for the ASI-Bench LLM judge.

Why: the ASI-Bench scorer (pinned commit) sends temperature 0.0 with reasoning_effort to openai/gpt-5.5, and the OpenAI API
rejects any temperature other than its default for that model, so every judge-scored task would come back unscored. The
maintainers' own documented setup routes the judge through a third-party gateway that tolerates the parameter. This proxy
plays that gateway role locally: it forwards each request to https://api.openai.com unchanged except that, for gpt-5* models,
the unsupported `temperature` is removed (also from `extra_body`). Model, prompt, reasoning effort and max tokens are untouched.

Usage (from the repository root; the ASI-Bench runner starts it automatically for local scoring):
    PYTHONPATH=src .venv/asibench/bin/python -m apex_eval.judge_proxy --port 8019 &
    scripts/evaluate.sh asibench ... --judge-api-base http://127.0.0.1:8019/v1 --judge-api-key-env OPENAI_API_KEY --judge-api-protocol openai
Inputs : OPENAI_API_KEY in the environment (the scorer sends it as the bearer token; the proxy forwards it as-is).
Outputs: one JSON line per forwarded request in --log (model, dropped params, HTTP status, usage), never the prompt or the key.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://api.openai.com"
FORWARDED_HEADERS = ("Authorization", "OpenAI-Organization", "OpenAI-Project")


class ProxyServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], strip_for: str, timeout: float, log_path: str):
        super().__init__(address, Handler)
        self.strip_for, self.upstream_timeout, self.log_path = strip_for, timeout, log_path


class Handler(BaseHTTPRequestHandler):
    server: ProxyServer

    def log_message(self, *args) -> None:  # the request log below replaces the default stderr access log
        pass

    def _record(self, rec: dict) -> None:
        with open(self.server.log_path, "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")

    def _forward(self, req: urllib.request.Request) -> tuple[int, bytes]:
        for h in FORWARDED_HEADERS:
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=self.server.upstream_timeout) as r:  # noqa: S310
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def _reply(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        dropped: list[str] = []
        model = "?"
        try:
            body = json.loads(raw)
            model = str(body.get("model", ""))
            if model.startswith(self.server.strip_for):
                if "temperature" in body:
                    body.pop("temperature")
                    dropped.append("temperature")
                extra = body.get("extra_body")
                if isinstance(extra, dict) and "temperature" in extra:
                    extra.pop("temperature")
                    dropped.append("extra_body.temperature")
                    if not extra:
                        body.pop("extra_body")
            raw = json.dumps(body).encode()
        except json.JSONDecodeError:
            pass   # forwarded unchanged; the upstream error is returned to the scorer
        req = urllib.request.Request(UPSTREAM + self.path, data=raw, method="POST", headers={"Content-Type": "application/json"})
        status, payload = self._forward(req)
        usage = None
        with contextlib.suppress(json.JSONDecodeError, AttributeError):
            usage = json.loads(payload).get("usage")
        self._record({"path": self.path, "model": model, "dropped": dropped, "status": status, "usage": usage})
        self._reply(status, payload)

    def do_GET(self) -> None:
        status, payload = self._forward(urllib.request.Request(UPSTREAM + self.path))
        self._reply(status, payload)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8019)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--strip-for", default="gpt-5", help="model-id prefix for which `temperature` is removed (default gpt-5)")
    p.add_argument("--timeout", type=float, default=600.0, help="upstream request timeout in seconds")
    p.add_argument("--log", default="judge_proxy_requests.jsonl")
    args = p.parse_args()
    print(f"judge proxy on {args.host}:{args.port} -> {UPSTREAM}; dropping temperature for models starting with {args.strip_for!r}; log {args.log}", flush=True)
    ProxyServer((args.host, args.port), args.strip_for, args.timeout, args.log).serve_forever()


if __name__ == "__main__":
    main()
