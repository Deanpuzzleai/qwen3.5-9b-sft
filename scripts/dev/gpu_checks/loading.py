"""Model / tokenizer loading, warning capture and synthetic chat data shared by the GPU checks.

Inputs : a checkpoint directory (the locked base snapshot on the mount, a merged save, or the tiny twin for CPU dry
         runs). Outputs: in-memory objects only; nothing is written here.

Conventions (configs/sft.yaml model:): the training/serving class is Qwen3_5ForConditionalGeneration (Qwen/Qwen3.5-9B:
dense hybrid decoder + vision tower), loaded with AutoModelForImageTextToText; attn_implementation="sdpa"; bf16
weights. The CPU dry runs use a shrunken twin of that class built by make_tiny_cg_twin.py (build_tiny_cg_config below).
"""
from __future__ import annotations

import inspect
import json
import logging
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = REPO_ROOT / "configs/eval/model_snapshot"
DEFAULT_MODEL_DIR = Path("/mnt/apex-shared/models/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a")   # model.yaml local_dir
TEST_TOKENIZER_DIR = REPO_ROOT / "logs/train_stack_check_qwen35_9b/tiny_base"   # tokenizer files of the tiny base used by the CPU checks
CG_CLASS = "Qwen3_5ForConditionalGeneration"
TEXT_ONLY_MODEL_TYPE = "qwen3_5_text"
MTP_RE = re.compile(r"^mtp\.")   # the checkpoint's 15 multi-token-prediction tensors (unused by the HF class)
FALLBACK_RE = re.compile(r"`(\w+)` is falling back to its reference PyTorch implementation")
IM_END_ID, THINK_ID, THINK_END_ID = 248046, 248068, 248069
AUTO_CLASSES = ("image-text-to-text", "causal-lm")


def seed_everything(seed: int) -> None:
    """Deterministic seeds for python, numpy (if present) and torch (CPU + all CUDA devices)."""
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(cpu_dry_run: bool) -> str:
    """'cuda' on a GPU box, 'cpu' for --cpu-dry-run; refuses to silently run a GPU check on CPU."""
    import torch

    if cpu_dry_run:
        return "cpu"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this check is GPU-only (use --cpu-dry-run for the CPU plumbing test)")
    return "cuda"


def load_tokenizer(model_dir: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


def auto_class_for(model_dir: str | Path, requested: str = "image-text-to-text") -> str:
    """Resolve the auto class: the served class for a qwen3_5 config, causal-lm only for a text-only twin."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    if requested == "image-text-to-text" and cfg.get("model_type") == TEXT_ONLY_MODEL_TYPE:
        return "causal-lm"
    return requested


def load_model(model_dir: str | Path, *, device: str, dtype_name: str = "bfloat16",
               auto_class: str = "image-text-to-text", attn_implementation: str = "sdpa") -> tuple[Any, dict[str, Any], float]:
    """from_pretrained with output_loading_info; returns (model, loading_info, seconds).

    loading_info = {"class", "missing", "unexpected", "unexpected_non_mtp", "mismatched", "weight_conversions",
                    "attn_implementation", "dtype", "n_params", "seconds"}.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    cls = {"image-text-to-text": AutoModelForImageTextToText, "causal-lm": AutoModelForCausalLM}[auto_class]
    dtype = getattr(torch, dtype_name)
    t0 = time.time()
    model, info = cls.from_pretrained(str(model_dir), dtype=dtype, device_map=device, output_loading_info=True,
                                      attn_implementation=attn_implementation)
    seconds = time.time() - t0
    unexpected = sorted(info.get("unexpected_keys", []))
    details = {
        "class": type(model).__name__,
        "missing": sorted(info.get("missing_keys", [])),
        "unexpected": unexpected,
        "unexpected_non_mtp": [k for k in unexpected if not MTP_RE.match(k)],
        "mismatched": [str(m) for m in info.get("mismatched_keys", [])],
        "weight_conversions": [str(c) for c in (getattr(model, "_weight_conversions", None) or [])],
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "dtype": str(next(model.parameters()).dtype),
        "n_params": sum(p.numel() for p in model.parameters()),
        "seconds": round(seconds, 1),
    }
    return model, details, seconds


class WarningCapture:
    """Collect transformers logger records and python warnings emitted inside the block.

    `.fallback_functions` lists the kernels that fell back to the reference PyTorch implementation
    (transformers/integrations/hub_kernels.py warning text), e.g. {"chunk_gated_delta_rule", "causal_conv1d_fn"}.
    """

    def __init__(self) -> None:
        self.records: list[str] = []
        self._handler = logging.Handler()
        self._handler.emit = lambda r: self.records.append(r.getMessage())  # type: ignore[method-assign]
        self._ctx: Any = None

    def __enter__(self) -> "WarningCapture":
        import transformers.utils.logging as tlog

        self._logger = tlog.get_logger()
        self._logger.addHandler(self._handler)
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.WARNING)
        self._ctx = warnings.catch_warnings(record=True)
        self._caught = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc: Any) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev_level)
        self.records.extend(str(w.message) for w in self._caught)
        self._ctx.__exit__(*exc)

    @property
    def fallback_functions(self) -> list[str]:
        return sorted({m.group(1) for r in self.records for m in [FALLBACK_RE.search(r)] if m})

    def grep(self, pattern: str) -> list[str]:
        rx = re.compile(pattern, re.IGNORECASE)
        return [r for r in self.records if rx.search(r)]


def build_tiny_cg_config(snapshot_config: Path = SNAPSHOT_DIR / "config.json"):
    """Shrunken Qwen3_5ForConditionalGeneration config with the 9B's structure: 4 decoder layers in the real pattern
    (3 Gated DeltaNet + 1 full attention, full_attention_interval 4), dense MLP, gated attention (attn_output_gate),
    a 1-block vision tower, the real 248,320 vocab, mamba_ssm_dtype float32 and the MTP fields kept; toy widths.
    Everything not listed below is copied from the snapshot's config.json (shape derived from the previous shrunken twin,
    tiny_config in probe 05v_verify_trl_condgen_cpu.py of the archived 35B training-stack study)."""
    from transformers import AutoConfig

    real = json.loads(Path(snapshot_config).read_text())
    tc = dict(real["text_config"])
    tc.update({
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"], "full_attention_interval": 4,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
        "linear_num_key_heads": 4, "linear_key_head_dim": 16, "linear_num_value_heads": 8, "linear_value_head_dim": 16,
        "max_position_embeddings": 4096, "use_cache": False,
        "rope_parameters": {**real["text_config"]["rope_parameters"], "mrope_section": [1, 1, 0]},   # rotary dim 4 at head_dim 16
    })
    vc = dict(real["vision_config"])
    vc.update({"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
               "num_position_embeddings": 16, "deepstack_visual_indexes": []})
    top = {k: v for k, v in real.items() if k not in ("text_config", "vision_config", "transformers_version")}
    top["text_config"], top["vision_config"] = tc, vc
    return AutoConfig.for_model(**top)


def text_config_of(model_dir: str | Path):
    """The text config of a checkpoint dir (top-level text_config for the served class, the config itself otherwise)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_dir))
    return getattr(cfg, "text_config", cfg)


# ---------------------------------------------------------------------------------- synthetic chats (own text)
def oai_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """OpenAI tool schema in the key order vLLM's pydantic models serialise (type -> function -> name, description,
    parameters), so HF renders match the server. Lifted from oai_tool() in
    logs/train_stack_check_qwen35_9b/probes/02_chat_template_format.py."""
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    oai_tool("run_shell", "Run a shell command in the sandbox and return stdout+stderr.",
             {"command": {"type": "string"}, "timeout_s": {"type": "integer", "default": 60}}, ["command"]),
    oai_tool("read_file", "Read a slice of a text file with line numbers.",
             {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    oai_tool("write_file", "Create or overwrite a text file.", {"path": {"type": "string"}, "content": {"type": "string"}},
             ["path", "content"]),
    oai_tool("search_code", "Search the repository for a regular expression.",
             {"pattern": {"type": "string"}, "max_results": {"type": "integer", "default": 50}}, ["pattern"]),
]


def tool_call(name: str, **arguments: Any) -> dict:
    """`arguments` stays a dict: the chat template raises TypeError on JSON strings (REPORT.md section 2)."""
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def synthetic_conversations() -> list[dict[str, Any]]:
    """Nine own conversations covering: single turn with thinking, tool episodes (one and two calls), parallel calls,
    two user queries (think-stripping branch), no-tools chat, long user text. Every assistant turn carries
    reasoning_content (REPORT.md: thinking-mode data must). Returns dicts {name, messages, tools, kwargs}. The locked
    template has no preserve_thinking variable (passing it is a silent no-op), so there is no preserve variant."""
    sysmsg = {"role": "system", "content": "You are a careful coding assistant working in a sandbox."}
    convs: list[dict[str, Any]] = []
    convs.append({"name": "single_turn", "tools": None, "kwargs": {}, "messages": [
        sysmsg, {"role": "user", "content": "Explain in two sentences why a hash map lookup is O(1) on average."},
        {"role": "assistant", "reasoning_content": "Hashing spreads keys over buckets; expected chain length is constant.",
         "content": "A hash function maps each key to a bucket in constant time, and with a good hash and load factor "
                    "each bucket holds a constant number of entries on average. Lookup therefore costs O(1) expected time."}]})
    convs.append({"name": "tool_episode", "tools": TOOLS[:2], "kwargs": {}, "messages": [
        sysmsg, {"role": "user", "content": "The test tests/test_parse.py::test_empty fails. Inspect src/parse.py and fix it."},
        {"role": "assistant", "reasoning_content": "Read the parser before changing anything.", "content": "Reading the file.",
         "tool_calls": [tool_call("read_file", path="src/parse.py", start_line=1, end_line=30)]},
        {"role": "tool", "name": "read_file", "content": "1 def parse(s):\n2     return s.split(',')[0]\n"},
        {"role": "assistant", "reasoning_content": "Empty input gives [''] so index 0 is '' which the test rejects; guard it.",
         "content": "", "tool_calls": [tool_call("run_shell", command="sed -i 's/return s.split/return None if not s else s.split/' src/parse.py && pytest -q tests/test_parse.py", timeout_s=120)]},
        {"role": "tool", "name": "run_shell", "content": "1 passed in 0.02s"},
        {"role": "assistant", "reasoning_content": "Passed; summarise.", "content": "Fixed: parse() now returns None for empty input; the test passes."}]})
    convs.append({"name": "parallel_calls", "tools": [TOOLS[3]], "kwargs": {}, "messages": [
        sysmsg, {"role": "user", "content": "Where is `load_config` defined and where is it used?"},
        {"role": "assistant", "reasoning_content": "Search definition and usages at once.", "content": "",
         "tool_calls": [tool_call("search_code", pattern=r"def load_config\(", max_results=5),
                        tool_call("search_code", pattern="load_config(", max_results=20)]},
        {"role": "tool", "name": "search_code", "content": "src/config.py:12:def load_config(path):"},
        {"role": "tool", "name": "search_code", "content": "src/app.py:3:from src.config import load_config\nsrc/app.py:40:    cfg = load_config(p)"},
        {"role": "assistant", "reasoning_content": "One definition, two uses.", "content": "Defined in src/config.py:12; used in src/app.py lines 3 and 40."}]})
    convs.append({"name": "two_queries", "tools": TOOLS[:1], "kwargs": {}, "messages": [
        sysmsg, {"role": "user", "content": "What is 19 * 21?"},
        {"role": "assistant", "reasoning_content": "19*21 = 399.", "content": "19 × 21 = 399."},
        {"role": "user", "content": "Now show the current directory listing."},
        {"role": "assistant", "reasoning_content": "Use run_shell with ls.", "content": "", "tool_calls": [tool_call("run_shell", command="ls -1")]},
        {"role": "tool", "name": "run_shell", "content": "README.md\nsrc\ntests"},
        {"role": "assistant", "reasoning_content": "Report it.", "content": "The directory holds README.md, src/ and tests/."}]})
    convs.append({"name": "nothink_render", "tools": TOOLS[:2], "kwargs": {"enable_thinking": False}, "messages": [
        sysmsg, {"role": "user", "content": "Create notes.txt containing the word hello."},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("write_file", path="notes.txt", content="hello\n")]},
        {"role": "tool", "name": "write_file", "content": "ok"},
        {"role": "assistant", "content": "Done: notes.txt written."}]})
    convs.append({"name": "plain_chat", "tools": None, "kwargs": {}, "messages": [
        {"role": "user", "content": "Give me a one-line bash command that counts lines in all .py files under src/."},
        {"role": "assistant", "reasoning_content": "find + xargs wc, or a glob with wc -l.",
         "content": "`find src -name '*.py' -print0 | xargs -0 cat | wc -l`"}]})
    long_user = " ".join(f"Record {i}: sensor {i % 7} reported {i * 3 % 101} units at tick {i}." for i in range(120))
    convs.append({"name": "long_user_text", "tools": None, "kwargs": {}, "messages": [
        sysmsg, {"role": "user", "content": long_user + "\n\nHow many records mention sensor 3?"},
        {"role": "assistant", "reasoning_content": "Indices with i % 7 == 3 in 0..119: 3,10,...,115 -> 17 records.",
         "content": "17 records mention sensor 3."}]})
    convs.append({"name": "unicode_and_code", "tools": None, "kwargs": {}, "messages": [
        {"role": "user", "content": "Write a Python function slugify('Crème Brûlée') -> 'creme-brulee'."},
        {"role": "assistant", "reasoning_content": "NFKD normalise, drop combining marks, lowercase, replace non-alnum runs with '-'.",
         "content": "```python\nimport re, unicodedata\n\ndef slugify(text: str) -> str:\n    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n```"}]})
    convs.append({"name": "tool_result_json", "tools": TOOLS[:1], "kwargs": {}, "messages": [
        sysmsg, {"role": "user", "content": "Run the unit tests and report the result."},
        {"role": "assistant", "reasoning_content": "pytest -q.", "content": "", "tool_calls": [tool_call("run_shell", command="pytest -q", timeout_s=600)]},
        {"role": "tool", "name": "run_shell", "content": json.dumps({"returncode": 1, "output": "FAILED tests/test_io.py::test_round_trip - AssertionError\n1 failed, 41 passed"})},
        {"role": "assistant", "reasoning_content": "One failure in test_io; report it.", "content": "41 passed, 1 failed: tests/test_io.py::test_round_trip (AssertionError)."}]})
    return convs


def render_ids(tok, messages: list[dict], tools: list[dict] | None, *, add_generation_prompt: bool,
               **kwargs: Any) -> list[int]:
    """Token ids of apply_chat_template(...) (tokenize=True) with the exact kwargs the server would receive."""
    out = tok.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=add_generation_prompt,
                                  return_dict=True, **kwargs)
    return list(out["input_ids"])


def read_prompts(path: Path, kind: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Own prompt set (prompts_own.jsonl): rows {id, kind: plain|tool, messages, tools?, expect_tool?}."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    return rows[:limit] if limit else rows


def callable_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the callable does not accept (what transformers' hub-kernel wrapper does before dispatch)."""
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
