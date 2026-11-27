#!/usr/bin/env python3
"""Opencode process smoke (data plan WP10, guard of section 1): ten hand-written data-analysis chores on generated CSV
files, each run once through opencode 1.17.15 against an OpenAI-compatible endpoint with the ASI-Bench runner's model
configuration. No scoring and no judge: the run records only process signals.

Usage (from the repository root):
  .venv/train/bin/python scripts/dev/opencode_smoke.py --api-base http://172.18.0.1:8002/v1 --model-name qwen35-base \
      [--version base|sft|stage2] [--prompts N] [--parallel K] [--timeout-sec S] [--baseline logs/<run_id of Base>]

Inputs : the endpoint (--api-base, --model-name; key from EVAL_API_KEY, default EMPTY, never logged); the opencode binary
         at ~/.npm-global/bin/opencode (1.17.15, mounted read-only into the sandbox); Docker; configs/eval/asibench.yaml
         (read-only: the tool mode and opencode version the configuration must match); --baseline: a previous run of this
         script (its directory or its summary.json). The chores and their CSV files are defined below and generated from
         --seed, so every model version gets byte-identical inputs (prompt_set_sha256 in the summary).
Outputs: logs/<run_id>/   metadata.json, eval.log, errors.log, resources.csv;
                          metrics.jsonl  the per-prompt table (one row per chore: finished, waiting_for_user, files_ok,
                                         parse_failures, tool_calls, turns, tokens, time_sec, run_status, ...);
                          summary.json   totals, the parse-failure rate and the guard;
                          prompts/<id>/  prompt.md, events.jsonl (opencode's JSON event stream), stderr.log, result.json
                                         (the row plus details), outputs/ (copies of the declared output files found)
         <root>/.cache/opencode_smoke/<run_id>/<id>/{workspace,home}   the sandbox's working directory and opencode's own
                          state (git-ignored; ~0.1 GB per chore, mostly the npm packages opencode installs into a fresh HOME
                          at start-up; kept for debugging, never deleted by this script)

Model configuration (identical to the ASI-Bench runner's `opencode_cli` adapter, as captured in
logs/train_stack_check/harness_request_shapes/captures/opencode_result.json): `opencode run --format json --auto
-m bench/<model>` with the prompt on stdin, and OPENCODE_CONFIG_CONTENT = one `@ai-sdk/openai-compatible` provider `bench`
whose baseURL is --api-base and whose only model is the served name, with no limit block (so opencode sends max_tokens
32000 per request and never compacts; the context is bounded by the server's max-model-len, 262144 in the lock), and with
webfetch and websearch disabled (the lock's tool_mode `restricted`). Through stdin the chore text reaches the model
verbatim as the user message (opencode wraps a positional message argument in quotes); it is also left in the workspace
as prompt.md. Sampling is the server's default (opencode sends no temperature and no seed).

Sandbox: one Docker container per chore, named ocsmoke_<run_id>_<id> and removed on exit: python:3.12-slim (pinned by
digest) with pandas, numpy, matplotlib and ripgrep (image ocsmoke-sandbox:<hash of SANDBOX_DOCKERFILE>, built once), the
host's uid, host networking (the endpoint is reached at the same address as from the host), no capabilities, CPU and
memory limits. Only the workspace, a fresh HOME and the opencode binary are mounted, so opencode sees no user-level
configuration, skills or instruction files from the host.

Per-prompt signals (parsed from the event stream; `parse_events` and `outcome`):
  turns           model calls in the session (step_start events; opencode's separate title request is not counted)
  tool_calls      tool_use events, plus tool calls that reached the text as raw markup
  parse_failures  calls opencode routed to its `invalid` tool (unparsable arguments, unknown tool name, schema mismatch),
                  tool errors that are argument-validation errors, and assistant text carrying raw tool-call markup
                  (<tool_call>, <function=...>: the server's parser did not turn it into a call)
  tool_errors     other failed tool executions (file not found, edit mismatch, ...); reported, not a parse failure
  finished        opencode exited on its own before the timeout with exit code 0, no session error, last step `stop`
  waiting_for_user  the final assistant text hands the turn back with a question or a request for input while a declared
                  output file is missing (an offer made after all files exist is not waiting)
  files_ok        every declared output file exists at its exact name in the workspace root and is non-empty
  tokens          sums over the step_finish events: input, output, reasoning, cache read
  time_sec        wall clock of the sandbox run, including container start-up (slow when Docker is busy);
                  agent_time_sec is the span from the first to the last event (the agent's working time)
Guard (data plan section 1, revision 4.3.11): parse_failure_rate = parse failures / tool calls over all chores (0 when
there are no calls); pass = rate <= 5 percent (exact fraction) and every run finished without waiting for a user. With
--baseline (same prompt_set_sha256) the comparison with the baseline's rate is reported next to it, not part of pass
(Base made no parse failure, so a relative bound would have no slack).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from apex_data.coordination import wait_for_docker_window  # noqa: E402
from apex_eval.asibench import LOCK as ASIBENCH_LOCK  # noqa: E402
from apex_eval.common import (AGENT_ERROR, BUDGET_EXHAUSTED, ENVIRONMENT_FAILURE, OK, VERSIONS, RunDir,  # noqa: E402
                              command_line, compute_rates, git_info, hardware_info, model_lock, new_run_id, preflight,
                              read_json, resource_rows, sha256_file, utc_now, write_json)

OPENCODE_BIN = Path.home() / ".npm-global" / "bin" / "opencode"
ADAPTER_CAPTURE = REPO_ROOT / "logs" / "train_stack_check" / "harness_request_shapes" / "captures" / "opencode_result.json"
SANDBOX_DOCKERFILE = """\
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas==3.0.6 numpy==2.5.3 matplotlib==3.11.2
"""
SANDBOX_IMAGE = "ocsmoke-sandbox:" + hashlib.sha256(SANDBOX_DOCKERFILE.encode()).hexdigest()[:12]
SANDBOX_LIMITS = ("--cpus", "2", "--memory", "4g", "--pids-limit", "1024")
CONTAINER_PREFIX = "ocsmoke_"
GUARD_MAX_RATE = Fraction(5, 100)
DEFAULT_SEED = 20261012
# prompt_set_sha256 of all ten chores at DEFAULT_SEED: the set of Base's reference run (logs/opencode_smoke_base_20261012_063939).
# Editing a chore, its generator or the seed makes later runs incomparable with that reference (the guard then fails).
REFERENCE_PROMPT_SET_SHA256 = "05caa10591c7ab8513dc4f76c35d0d606f70bdb0e12a7ed569eee364bbcf096e"
OUTPUT_COPY_MAX_BYTES = 5_000_000

# --- the ten chores (hand-written; synthetic data from a seeded generator) --------------------------------------------
Rows = list[list[Any]]


@dataclass(frozen=True)
class Chore:
    id: str
    title: str
    inputs: Callable[[random.Random], dict[str, Rows]]   # file name -> rows, header first
    outputs: tuple[str, ...]
    task: str

    def prompt(self, input_names: list[str]) -> str:
        files = ", ".join(input_names)
        return (f"{self.task}\n\nThe input file{'s' if len(input_names) > 1 else ''} ({files}) {'are' if len(input_names) > 1 else 'is'} in the "
                f"current working directory. Write the output file{'s' if len(self.outputs) > 1 else ''} there as well, with exactly "
                f"{'these names' if len(self.outputs) > 1 else 'this name'}: {', '.join(self.outputs)}.")


def _days(r: random.Random, start: date, n: int) -> str:
    return (start + timedelta(days=r.randrange(n))).isoformat()


def gen_sales(r: random.Random) -> dict[str, Rows]:
    price = {"A": 12.5, "B": 19.99, "C": 7.25, "D": 45.0, "E": 99.9}
    rows: Rows = [["order_date", "region", "product", "units", "unit_price"]]
    for _ in range(600):
        p = r.choice("ABCDE")
        rows.append([_days(r, date(2025, 1, 1), 365), r.choice(["North", "South", "East", "West"]), p, r.randint(1, 20),
                     f"{price[p] * r.uniform(0.9, 1.1):.2f}"])
    return {"sales.csv": rows}


def gen_customers_orders(r: random.Random) -> dict[str, Rows]:
    countries = ["Germany", "France", "Spain", "Italy", "Poland", "Sweden", "Portugal", "Austria"]
    first = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jo"]
    last = ["Berg", "Costa", "Diaz", "Engel", "Fischer", "Novak", "Rossi", "Silva"]
    customers: Rows = [["customer_id", "name", "country"]]
    for i in range(1, 61):
        customers.append([f"C{i:03d}", f"{r.choice(first)} {r.choice(last)}", r.choice(countries)])
    orders: Rows = [["order_id", "customer_id", "amount", "status"]]
    for i in range(1, 401):
        cid = f"C{r.randint(1, 60):03d}" if r.random() > 0.05 else f"C{r.randint(900, 999)}"
        status = r.choices(["completed", "cancelled", "refunded", "pending"], weights=[70, 12, 8, 10])[0]
        orders.append([f"O{i:04d}", cid, f"{r.lognormvariate(4.0, 0.6):.2f}", status])
    return {"customers.csv": customers, "orders.csv": orders}


def gen_sensors(r: random.Random) -> dict[str, Rows]:
    variants = {"north-1": ["north-1", "North-1", "  north-1", "NORTH-1 "], "north-2": ["north-2", "North-2 ", " NORTH-2"],
                "south-1": ["south-1", "South-1", " south-1  "]}
    body: Rows = []
    t0 = datetime(2025, 6, 1)
    for station, names in variants.items():
        for h in range(72):
            temp: Any = f"{18 + 6 * r.random() + (2 if station == 'south-1' else 0):.1f}"
            if r.random() < 0.06:
                temp = r.choice(["", "n/a"])
            hum = r.randint(35, 90)
            body.append([r.choice(names), (t0 + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M"), temp,
                         f"{hum}%" if r.random() < 0.3 else str(hum)])
    for row in r.sample(body, 20):
        body.insert(r.randrange(len(body)), list(row))
    return {"sensors_raw.csv": [["station", "timestamp", "temperature_c", "humidity_pct"], *body]}


def gen_measurements(r: random.Random) -> dict[str, Rows]:
    mu = {"control": 10.0, "treatment_a": 11.5, "treatment_b": 9.2}
    rows: Rows = [["sample_id", "group", "value"]]
    for i in range(1, 301):
        g = r.choice(list(mu))
        rows.append([f"S{i:04d}", g, f"{r.gauss(mu[g], 2.0):.3f}"])
    return {"measurements.csv": rows}


def gen_daily_temps(r: random.Random) -> dict[str, Rows]:
    base = {"Oslo": (6.0, 10.0), "Madrid": (15.0, 9.0), "Cairo": (22.5, 7.0)}
    rows: Rows = [["date", "city", "temp_c"]]
    for d in range(365):
        for city, (mean, amp) in base.items():
            season = -math.cos(2 * math.pi * (d - 15) / 365)
            rows.append([(date(2025, 1, 1) + timedelta(days=d)).isoformat(), city, f"{mean + amp * season + r.gauss(0, 2.5):.1f}"])
    return {"daily_temps.csv": rows}


def gen_employees(r: random.Random) -> dict[str, Rows]:
    pay = {"Engineering": 92000, "Sales": 64000, "Support": 48000, "Finance": 71000}
    rows: Rows = [["employee_id", "department", "hire_date", "salary"]]
    for i in range(1, 251):
        dept = r.choices(list(pay), weights=[40, 25, 20, 15])[0]
        rows.append([f"E{i:04d}", dept, _days(r, date(2012, 1, 1), 5000), int(r.gauss(pay[dept], pay[dept] * 0.15) // 100 * 100)])
    return {"employees.csv": rows}


def gen_survey(r: random.Random) -> dict[str, Rows]:
    groups = ["18-29", "30-44", "45-59", "60+"]
    rows: Rows = [["respondent_id", "age_group", "question", "score"]]
    for i in range(1, 121):
        g = r.choice(groups)
        for q in range(1, 6):
            score = min(5, max(1, round(r.gauss(3 + 0.3 * groups.index(g) - 0.2 * q, 1.0))))
            rows.append([f"R{i:03d}", g, f"Q{q}", "" if r.random() < 0.03 else score])
    return {"survey.csv": rows}


def gen_transactions(r: random.Random) -> dict[str, Rows]:
    accounts = {f"ACC{k}": (r.uniform(20, 200), r.uniform(5, 30)) for k in range(1, 7)}
    body: Rows = []
    for _ in range(500):
        acc = r.choice(list(accounts))
        mean, sd = accounts[acc]
        body.append([acc, max(0.5, r.gauss(mean, sd))])
    for acc in r.sample(list(accounts), 4):
        mean, sd = accounts[acc]
        body.insert(r.randrange(len(body)), [acc, mean + sd * r.uniform(8, 12)])
    return {"transactions.csv": [["txn_id", "account", "amount"], *[[f"T{i:05d}", a, f"{x:.2f}"] for i, (a, x) in enumerate(body, 1)]]}


def gen_exam(r: random.Random) -> dict[str, Rows]:
    rows: Rows = [["student_id", "class", "score"]]
    for i in range(1, 151):
        c = r.choice("ABC")
        rows.append([f"ST{i:03d}", c, min(100, max(0, round(r.gauss({"A": 68, "B": 61, "C": 74}[c], 14))))])
    return {"exam_scores.csv": rows}


def gen_meter(r: random.Random) -> dict[str, Rows]:
    rows: Rows = [["timestamp", "kwh"]]
    t0 = datetime(2025, 3, 1)
    for h in range(14 * 24):
        if r.random() < 0.05:
            continue
        load = 0.4 + 0.5 * max(0.0, math.sin(math.pi * ((h % 24) - 6) / 14)) + r.uniform(0, 0.2)
        rows.append([(t0 + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"), f"{load:.3f}"])
    return {"meter_readings.csv": rows}


CHORES: tuple[Chore, ...] = (
    Chore("01_monthly_revenue", "Monthly revenue from an order list", gen_sales, ("monthly_revenue.csv",),
          "sales.csv contains one row per order (columns: order_date, region, product, units, unit_price). Calculate the total "
          "revenue of each calendar month, where an order's revenue is units times unit_price. Save the result as "
          "monthly_revenue.csv with two columns, month (formatted YYYY-MM) and revenue (rounded to 2 decimal places), one row per "
          "month in chronological order."),
    Chore("02_country_join", "Completed-order totals by country (join)", gen_customers_orders, ("revenue_by_country.csv",),
          "customers.csv lists customers (customer_id, name, country) and orders.csv lists orders (order_id, customer_id, amount, "
          "status). Using only the orders whose status is completed, compute for each country the number of orders and their total "
          "amount; leave out orders whose customer_id does not appear in customers.csv. Write revenue_by_country.csv with the "
          "columns country, n_orders and total_amount (rounded to 2 decimals), sorted by total_amount from highest to lowest."),
    Chore("03_clean_sensors", "Clean a messy sensor export", gen_sensors, ("sensors_clean.csv",),
          "sensors_raw.csv is a messy export from weather stations (columns: station, timestamp, temperature_c, humidity_pct). "
          "Clean it: trim surrounding whitespace from the station names and convert them to lower case; remove the % sign from "
          "humidity values so that the column is numeric; treat empty temperature values and the text n/a as missing and drop those "
          "rows; then remove duplicate rows. Sort by station and then by timestamp and write the result to sensors_clean.csv with "
          "the same four columns in the same order."),
    Chore("04_summary_json", "Summary statistics as JSON", gen_measurements, ("summary.json",),
          "measurements.csv has the columns sample_id, group and value. Produce summary.json, a JSON object with these keys: "
          "n_rows (an integer), mean, median, std (the sample standard deviation), min and max of value, and group_means (an "
          "object mapping each group name to the mean value of that group). Round all floating-point numbers to 4 decimal places."),
    Chore("05_rolling_plot", "Rolling-mean temperature plot", gen_daily_temps, ("rolling_temps.png",),
          "daily_temps.csv holds one temperature reading per city per day of 2025 (columns: date, city, temp_c). Plot the 7-day "
          "rolling mean of temp_c for each city as a line chart, with all cities on the same axes, a legend, labelled axes and a "
          "title, and save the figure as rolling_temps.png."),
    Chore("06_filter_engineers", "Filter and rank employees", gen_employees, ("selected_engineers.csv",),
          "employees.csv has the columns employee_id, department, hire_date and salary. Find the employees of the Engineering "
          "department who were hired on or after 2020-01-01 and whose salary is above the median salary of all Engineering "
          "employees. Write them to selected_engineers.csv with the columns employee_id, hire_date and salary, sorted by salary "
          "from highest to lowest."),
    Chore("07_survey_pivot", "Survey pivot table", gen_survey, ("score_pivot.csv",),
          "survey.csv contains survey answers in long format (columns: respondent_id, age_group, question, score) with scores from "
          "1 to 5; a few scores are blank and should be ignored. Build a table with one row per age group, in the order 18-29, "
          "30-44, 45-59, 60+, and one column per question, Q1 to Q5, holding the mean score rounded to 2 decimals. Save it as "
          "score_pivot.csv; its first column must be named age_group."),
    Chore("08_outliers", "Flag outlier transactions", gen_transactions, ("outliers.csv",),
          "transactions.csv lists card transactions (columns: txn_id, account, amount). For every transaction compute a z-score "
          "within its own account, using that account's mean amount and sample standard deviation. Write the transactions whose "
          "absolute z-score is greater than 3 to outliers.csv with the columns txn_id, account, amount and z_score (rounded to 2 "
          "decimals), sorted by absolute z-score from largest to smallest."),
    Chore("09_exam_report", "Exam histogram and Markdown report", gen_exam, ("score_hist.png", "report.md"),
          "exam_scores.csv has the columns student_id, class and score (0 to 100). Create two files: score_hist.png, a histogram "
          "of all scores with bins of width 10 from 0 to 100 and labelled axes; and report.md, a short Markdown report stating the "
          "number of students, the mean score of each class (2 decimals) and the overall pass rate as a percentage with one "
          "decimal, where a score of 60 or more is a pass."),
    Chore("10_daily_usage", "Hourly meter readings to daily usage", gen_meter, ("daily_usage.csv",),
          "meter_readings.csv contains hourly electricity meter readings (columns: timestamp, kwh); some hours are missing. "
          "Produce daily_usage.csv with one row per calendar day from the first to the last day in the data and the columns date "
          "(YYYY-MM-DD), total_kwh (the sum of that day's readings, rounded to 2 decimals) and missing_hours (24 minus the "
          "number of readings on that day)."),
)


def csv_bytes(rows: Rows) -> bytes:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue().encode()


def materialize(chore: Chore, seed: int) -> tuple[dict[str, bytes], str]:
    """The chore's input files (name -> bytes) and its full prompt; deterministic in (seed, chore id)."""
    files = {name: csv_bytes(rows) for name, rows in chore.inputs(random.Random(f"{seed}:{chore.id}")).items()}
    return files, chore.prompt(list(files))


def prompt_set_sha256(items: list[tuple[Chore, dict[str, bytes], str]]) -> str:
    spec = [{"id": c.id, "prompt": p, "outputs": list(c.outputs), "inputs": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
            for c, files, p in items]
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# --- model configuration (the ASI-Bench opencode_cli adapter's) -------------------------------------------------------
def asibench_opencode_settings() -> dict[str, Any]:
    """The lock values this smoke must match: opencode version and tool mode (configs/eval/asibench.yaml, read-only)."""
    lock = yaml.safe_load(ASIBENCH_LOCK.read_text())
    rm = lock["run_mode"]
    m = re.search(r"opencode (\d+\.\d+\.\d+)", str(rm.get("agent_version", "")))
    return {"agent": rm["agent"], "agent_config": rm["agent_config"], "tool_mode": rm["tool_mode"],
            "opencode_version": m.group(1) if m else None, "inference": lock["inference"]}


def opencode_config(api_base: str, model_name: str, api_key: str, tool_mode: str) -> dict[str, Any]:
    """OPENCODE_CONFIG_CONTENT as the adapter builds it (see the module docstring for the capture it reproduces)."""
    if tool_mode != "restricted":
        raise SystemExit(f"the ASI-Bench lock's tool_mode is {tool_mode!r}; this smoke reproduces only 'restricted' (as captured)")
    return {"provider": {"bench": {"npm": "@ai-sdk/openai-compatible", "name": "ASI-Bench Target",
                                   "options": {"baseURL": api_base, "apiKey": api_key},
                                   "models": {model_name: {"name": model_name}}}},
            "agent": {"build": {"tools": {"webfetch": False, "websearch": False}}}}


def opencode_args(model_name: str) -> list[str]:
    """The adapter's command line; the prompt goes to stdin (opencode reads a non-TTY stdin as the message, verbatim)."""
    return ["opencode", "run", "--format", "json", "--auto", "-m", f"bench/{model_name}"]


def redacted(config: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(config))
    opts = out["provider"]["bench"]["options"]
    if opts.get("apiKey") not in (None, "", "EMPTY"):
        opts["apiKey"] = "***"
    return out


# --- event-stream parsing (pure functions; tested on synthetic streams) ----------------------------------------------
ARGUMENT_ERROR = re.compile(r"called with invalid arguments|invalid input for tool|tried to call unavailable tool|"
                            r"arguments provided to the tool are invalid|json parse|not valid json|unexpected (?:token|end of json)",
                            re.IGNORECASE)
ASKS_USER = re.compile(r"\b(?:would you like|do you want|shall i|should i|could you (?:please )?(?:provide|confirm|clarify|specify|share)|"
                       r"can you (?:please )?(?:provide|confirm|clarify|specify|share)|please (?:confirm|clarify|specify|provide|share)|"
                       r"let me know (?:which|whether|how)|before i (?:proceed|continue|start))\b", re.IGNORECASE)


def parse_events(text: str) -> dict[str, Any]:
    """Counts from opencode's `run --format json` stream (one JSON object per line: step_start, text, reasoning, tool_use,
    step_finish, error). Lines that are not JSON objects are counted as noise, never fatal."""
    turns = tool_calls = noise = 0
    by_tool: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    tool_errors: list[dict[str, Any]] = []
    errors: list[Any] = []
    tokens = Counter({"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0})
    last_reason: str | None = None
    step_text: list[str] = []
    stamps: list[int] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            noise += 1
            continue
        if not isinstance(ev, dict):
            noise += 1
            continue
        kind, part = ev.get("type"), ev.get("part") or {}
        if isinstance(ev.get("timestamp"), int):
            stamps.append(ev["timestamp"])
        if kind == "step_start":
            turns += 1
            step_text = []
        elif kind == "step_finish":
            last_reason = part.get("reason")
            tk = part.get("tokens") or {}
            cache = tk.get("cache") or {}
            for key, val in (("input", tk.get("input")), ("output", tk.get("output")), ("reasoning", tk.get("reasoning")),
                             ("cache_read", cache.get("read")), ("cache_write", cache.get("write"))):
                tokens[key] += int(val or 0)
        elif kind == "tool_use":
            tool_calls += 1
            name = str(part.get("tool"))
            by_tool[name] += 1
            state = part.get("state") or {}
            if name == "invalid":
                inp = state.get("input") or {}
                failures.append({"kind": "invalid_tool", "tool": inp.get("tool"), "detail": str(inp.get("error") or state.get("output"))[:300]})
            elif state.get("status") == "error":
                msg = str(state.get("error"))
                if ARGUMENT_ERROR.search(msg):
                    failures.append({"kind": "invalid_arguments", "tool": name, "detail": msg[:300]})
                else:
                    tool_errors.append({"tool": name, "detail": msg[:300]})
        elif kind == "text":
            body = str(part.get("text") or "")
            step_text.append(body)
            n = len(re.findall(r"<function=", body)) or body.count("<tool_call>")   # one leaked call per function tag
            if n:
                tool_calls += n
                failures.extend({"kind": "markup_in_text", "tool": None, "detail": body[:300]} for _ in range(n))
        elif kind == "error":
            err = ev.get("error")
            errors.append(err if isinstance(err, (dict, str)) else str(err))
    final_text = "\n".join(t for t in step_text if t.strip()).strip()
    return {"turns": turns, "tool_calls": tool_calls, "tool_calls_by_name": dict(by_tool), "parse_failures": len(failures),
            "parse_failure_details": failures, "tool_errors": len(tool_errors), "tool_error_details": tool_errors,
            "session_errors": errors, "last_step_reason": last_reason, "final_text": final_text, "noise_lines": noise,
            "tokens": dict(tokens), "event_span_sec": round((max(stamps) - min(stamps)) / 1000, 1) if stamps else None}


def asks_user(final_text: str) -> bool:
    """True when the final assistant text hands the turn back with a question or a request for input."""
    t = final_text.strip()
    return bool(t) and (t.endswith("?") or bool(ASKS_USER.search(t)))


def check_outputs(workspace: Path, outputs: tuple[str, ...]) -> dict[str, Any]:
    """Declared output files at their exact names in the workspace root: present and non-empty (files_ok), plus a light
    well-formedness check by extension (CSV with a header and a row, JSON that loads, PNG signature, non-empty text).
    Well-formedness is informational; nothing is scored."""
    present, missing, wellformed = [], [], {}
    names = {p.name for p in workspace.iterdir()} if workspace.exists() else set()
    for name in outputs:
        f = workspace / name
        if name not in names or not f.is_file() or f.stat().st_size == 0:
            missing.append(name)
            continue
        present.append(name)
        data = f.read_bytes()
        try:
            if name.endswith(".csv"):
                rows = list(csv.reader(io.StringIO(data.decode())))
                ok = len(rows) >= 2 and len(rows[0]) >= 1
            elif name.endswith(".json"):
                json.loads(data)
                ok = True
            elif name.endswith(".png"):
                ok = data.startswith(b"\x89PNG\r\n\x1a\n")
            else:
                ok = bool(data.decode().strip())
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error):
            ok = False
        wellformed[name] = ok
    return {"files_ok": not missing, "present": present, "missing": missing, "wellformed": wellformed}


def outcome(events: dict[str, Any], files: dict[str, Any], *, exit_code: int | None, timed_out: bool, started: bool) -> dict[str, Any]:
    """The per-prompt verdicts from the parsed stream, the file check and the process result (no scoring)."""
    finished = (not timed_out and exit_code == 0 and not events["session_errors"] and events["last_step_reason"] == "stop")
    final_asks = asks_user(events["final_text"])
    waiting = final_asks and not files["files_ok"]
    if not started:
        status, reason = ENVIRONMENT_FAILURE, f"sandbox did not start (exit {exit_code})"
    elif timed_out:
        status, reason = BUDGET_EXHAUSTED, "timeout"
    elif not finished:
        errs = "; ".join(json.dumps(e)[:200] for e in events["session_errors"])
        status, reason = AGENT_ERROR, f"exit {exit_code}, last step {events['last_step_reason']!r}" + (f", session error: {errs}" if errs else "")
    else:
        status, reason = OK, None
    return {"run_status": status, "error_reason": reason, "finished": finished, "final_asks_user": final_asks,
            "waiting_for_user": waiting, "finished_without_user": finished and not waiting}


# --- summary and guard -------------------------------------------------------------------------------------------------
def rate(failures: int, calls: int) -> Fraction:
    return Fraction(failures, calls) if calls else Fraction(0)


def summarize(rows: list[dict[str, Any]], prompt_sha: str) -> dict[str, Any]:
    n = len(rows)
    calls = sum(r["tool_calls"] for r in rows)
    fails = sum(r["parse_failures"] for r in rows)
    times = [r["time_sec"] for r in rows if r.get("time_sec") is not None]
    agent = [r["agent_time_sec"] for r in rows if r.get("agent_time_sec") is not None]
    return {
        "n_prompts": n, "prompt_set_sha256": prompt_sha,
        "n_finished": sum(r["finished"] for r in rows), "n_finished_without_user": sum(r["finished_without_user"] for r in rows),
        "n_waiting_for_user": sum(r["waiting_for_user"] for r in rows), "n_timed_out": sum(r["timed_out"] for r in rows),
        "n_files_ok": sum(r["files_ok"] for r in rows),
        "tool_calls": calls, "parse_failures": fails, "parse_failure_rate_pct": round(100 * float(rate(fails, calls)), 3),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "turns_total": sum(r["turns"] for r in rows), "turns_mean": round(sum(r["turns"] for r in rows) / n, 2) if n else None,
        "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
        "reasoning_tokens": sum(r["reasoning_tokens"] for r in rows),
        "time_sec_mean": round(sum(times) / len(times), 1) if times else None, "time_sec_max": max(times) if times else None,
        "agent_time_sec_mean": round(sum(agent) / len(agent), 1) if agent else None,
        "run_status_counts": dict(Counter(r["run_status"] for r in rows)),
    }


def guard(summary: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Data plan section 1 (revision 4.3.11): parse failures <= 5 percent of tool calls and every run finishes without
    waiting for a user. The baseline's rate is reported next to it (informational, not part of pass)."""
    own = rate(summary["parse_failures"], summary["tool_calls"])
    checks = {"parse_failure_rate_le_5pct": own <= GUARD_MAX_RATE,
              "all_finished_without_user": summary["n_finished_without_user"] == summary["n_prompts"]}
    out: dict[str, Any] = {"rule": "parse_failures/tool_calls <= 5%; every run finished without waiting for a user",
                           "parse_failure_rate": f"{summary['parse_failures']}/{summary['tool_calls']}", "checks": checks,
                           "pass": all(checks.values()), "baseline": None}
    if baseline is not None:
        comparable = (baseline.get("prompt_set_sha256") == summary["prompt_set_sha256"]
                      and baseline.get("n_prompts") == summary["n_prompts"])
        base = rate(int(baseline.get("parse_failures") or 0), int(baseline.get("tool_calls") or 0))
        out["baseline"] = {"run_id": baseline.get("run_id"), "parse_failures": baseline.get("parse_failures"),
                           "tool_calls": baseline.get("tool_calls"), "parse_failure_rate_pct": baseline.get("parse_failure_rate_pct"),
                           "comparable": comparable, "rate_le_baseline": bool(comparable and own <= base),
                           "note": "informational: not part of the guard (revision 4.3.11)"}
    return out


def load_baseline(path: Path) -> dict[str, Any]:
    f = path / "summary.json" if path.is_dir() else path
    data = read_json(f)
    if not isinstance(data, dict) or "prompt_set_sha256" not in data:
        raise SystemExit(f"--baseline {path}: no summary.json of an opencode smoke run there")
    return data


# --- sandbox ------------------------------------------------------------------------------------------------------------
class Sandboxes:
    """Names of this run's live containers, so an interrupted run kills exactly its own and nothing else."""

    def __init__(self) -> None:
        self.live: set[str] = set()
        self.lock = threading.Lock()

    def add(self, name: str) -> None:
        with self.lock:
            self.live.add(name)

    def discard(self, name: str) -> None:
        with self.lock:
            self.live.discard(name)

    def kill_all(self) -> None:
        with self.lock:
            names = sorted(self.live)
        for name in names:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)


def ensure_image(run: RunDir) -> str:
    if subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE], capture_output=True, check=False).returncode == 0:
        return SANDBOX_IMAGE
    wait_for_docker_window(f"docker build {SANDBOX_IMAGE}", run.log)   # no build while a reportable Terminal-Bench run is active
    run.log(f"building {SANDBOX_IMAGE} (log: sandbox_build.log)")
    with (run.run_dir / "sandbox_build.log").open("w") as log:
        rc = subprocess.run(["docker", "build", "-t", SANDBOX_IMAGE, "-"], input=SANDBOX_DOCKERFILE.encode(), stdout=log,
                            stderr=subprocess.STDOUT, check=False).returncode
    if rc != 0:
        run.error(f"sandbox image build failed (exit {rc}); see sandbox_build.log")
        raise SystemExit(f"sandbox image build failed (exit {rc})")
    return SANDBOX_IMAGE


@dataclass
class Context:
    run: RunDir
    run_id: str
    work_root: Path
    image: str
    model_name: str
    config_json: str
    timeout: int
    sandboxes: Sandboxes


def run_chore(ctx: Context, chore: Chore, files: dict[str, bytes], prompt: str) -> dict[str, Any]:
    pdir = ctx.run.run_dir / "prompts" / chore.id
    work = ctx.work_root / chore.id
    ws, home = work / "workspace", work / "home"
    for d in (pdir, ws, home):
        d.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (ws / name).write_bytes(data)
    (ws / "prompt.md").write_text(prompt)
    (pdir / "prompt.md").write_text(prompt)
    name = f"{CONTAINER_PREFIX}{ctx.run_id}_{chore.id}"
    cmd = ["docker", "run", "--rm", "-i", "--init", "--name", name, "--label", f"ocsmoke.run_id={ctx.run_id}", "--network", "host",
           "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", *SANDBOX_LIMITS,
           "-v", f"{ws}:/workspace", "-v", f"{home}:/home/agent", "-v", f"{OPENCODE_BIN.resolve()}:/usr/local/bin/opencode:ro",
           "-w", "/workspace", "-e", "HOME=/home/agent", "-e", "OPENCODE_DISABLE_AUTOUPDATE=1", "-e", "OPENCODE_CONFIG_CONTENT",
           ctx.image, *opencode_args(ctx.model_name)]
    env = {**os.environ, "OPENCODE_CONFIG_CONTENT": ctx.config_json}   # passed by name: the key never reaches a command line
    events_path, stderr_path = pdir / "events.jsonl", pdir / "stderr.log"
    ctx.run.log(f"{chore.id}: start ({name})")
    t0 = time.time()
    timed_out = False
    with events_path.open("wb") as out, stderr_path.open("wb") as err:
        ctx.sandboxes.add(name)
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, env=env)
            try:
                proc.stdin.write(prompt.encode())
                proc.stdin.close()
            except BrokenPipeError:   # the sandbox exited before reading the prompt; its exit code tells why
                pass
            try:
                rc: int | None = proc.wait(timeout=ctx.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(["docker", "kill", name], capture_output=True, check=False)
                try:
                    rc = proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
        finally:
            ctx.sandboxes.discard(name)
    elapsed = round(time.time() - t0, 1)
    events = parse_events(events_path.read_text(errors="replace"))
    # docker's own failures (125-127) with an empty stream mean opencode never ran
    started = not (rc in (125, 126, 127) and events["turns"] == 0 and events["noise_lines"] == 0)
    check = check_outputs(ws, chore.outputs)
    for fname in check["present"]:
        src = ws / fname
        if src.stat().st_size <= OUTPUT_COPY_MAX_BYTES:
            (pdir / "outputs").mkdir(exist_ok=True)
            shutil.copy2(src, pdir / "outputs" / fname)
    verdict = outcome(events, check, exit_code=rc, timed_out=timed_out, started=started)
    tk = events["tokens"]
    row = {"prompt_id": chore.id, "title": chore.title, "run_status": verdict["run_status"], "finished": verdict["finished"],
           "waiting_for_user": verdict["waiting_for_user"], "finished_without_user": verdict["finished_without_user"],
           "files_ok": check["files_ok"], "missing_files": check["missing"], "parse_failures": events["parse_failures"],
           "tool_calls": events["tool_calls"], "tool_errors": events["tool_errors"], "turns": events["turns"],
           "input_tokens": tk["input"], "output_tokens": tk["output"], "reasoning_tokens": tk["reasoning"],
           "cache_read_tokens": tk["cache_read"], "time_sec": elapsed, "agent_time_sec": events["event_span_sec"], "exit_code": rc,
           "timed_out": timed_out, "last_step_reason": events["last_step_reason"], "error_reason": verdict["error_reason"]}
    known = {*files, "prompt.md", *chore.outputs}
    other = sorted(rel for rel in (str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()) if rel not in known)
    detail = {**row, "final_asks_user": verdict["final_asks_user"], "outputs_wellformed": check["wellformed"],
              "tool_calls_by_name": events["tool_calls_by_name"], "parse_failure_details": events["parse_failure_details"],
              "tool_error_details": events["tool_error_details"], "session_errors": events["session_errors"],
              "final_text": events["final_text"][-2000:], "noise_lines": events["noise_lines"], "tokens": tk,
              "other_files_written": other[:50], "n_other_files_written": len(other),
              "inputs_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
              "container": name, "workspace": str(ws), "opencode_home": str(home)}
    write_json(pdir / "result.json", detail)
    if verdict["run_status"] != OK:
        ctx.run.error(f"{chore.id}: {verdict['run_status']}: {verdict['error_reason']}")
    for f in events["parse_failure_details"]:
        ctx.run.error(f"{chore.id}: tool-call parse failure ({f['kind']}, tool={f['tool']}): {f['detail'][:200]}")
    ctx.run.log(f"{chore.id}: {verdict['run_status']} in {elapsed}s; turns {events['turns']}, tool calls {events['tool_calls']}, "
                f"parse failures {events['parse_failures']}, files_ok {check['files_ok']}, waiting_for_user {verdict['waiting_for_user']}")
    return row


def crashed_row(chore: Chore, exc: BaseException) -> dict[str, Any]:
    """A chore whose harness step raised is still recorded (never dropped)."""
    return {"prompt_id": chore.id, "title": chore.title, "run_status": ENVIRONMENT_FAILURE, "finished": False,
            "waiting_for_user": False, "finished_without_user": False, "files_ok": False, "missing_files": list(chore.outputs),
            "parse_failures": 0, "tool_calls": 0, "tool_errors": 0, "turns": 0, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "cache_read_tokens": 0, "time_sec": None, "agent_time_sec": None, "exit_code": None, "timed_out": False,
            "last_step_reason": None, "error_reason": f"harness error: {type(exc).__name__}: {exc}"[:500]}


def table(rows: list[dict[str, Any]]) -> str:
    head = (f"{'prompt':<22} {'fin':>3} {'wait':>4} {'files':>5} {'pf/calls':>8} {'turns':>5} {'in_tok':>9} {'out_tok':>8} {'reas_tok':>8} "
            f"{'sec':>6} {'agent_s':>7}  status")
    lines = [head]
    for r in rows:
        sec, agent = (r.get(k) if r.get(k) is not None else "-" for k in ("time_sec", "agent_time_sec"))
        lines.append(f"{r['prompt_id']:<22} {'y' if r['finished'] else 'n':>3} {'y' if r['waiting_for_user'] else 'n':>4} "
                     f"{'ok' if r['files_ok'] else 'MISS':>5} {str(r['parse_failures']) + '/' + str(r['tool_calls']):>8} {r['turns']:>5} "
                     f"{r['input_tokens']:>9} {r['output_tokens']:>8} {r['reasoning_tokens']:>8} {sec:>6} {agent:>7}  {r['run_status']}")
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    rates = compute_rates()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-base", required=True, help="OpenAI-compatible base URL as reachable from apex-cpu, e.g. http://172.18.0.1:8002/v1")
    p.add_argument("--model-name", required=True, help="served model id on the endpoint")
    p.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", "EMPTY"), help="endpoint key (default: EVAL_API_KEY or EMPTY)")
    p.add_argument("--version", default=None, choices=VERSIONS, help="model version label stored with the run (and in the default run id)")
    p.add_argument("--prompts", type=int, default=len(CHORES), help=f"run the first N chores (default {len(CHORES)}; fewer is not guard-comparable)")
    p.add_argument("--parallel", type=int, default=4, help="concurrent sandboxes (default 4)")
    p.add_argument("--timeout-sec", type=int, default=1800, help="per-chore wall-clock limit; a run still going is killed and counts as not finished")
    p.add_argument("--baseline", type=Path, default=None, help="a previous run of this script (logs/<run_id> or its summary.json) for the guard")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="generator seed of the CSV inputs (keep the default for comparable runs)")
    p.add_argument("--run-id", default=None, help="default: opencode_smoke[_<version>]_<UTC timestamp>")
    p.add_argument("--serving-gpus", type=int, default=rates["default_serving_gpus"], help="GPUs serving the model (resources.csv; 0 for a mock)")
    p.add_argument("--peak-gpu-mem-gb", type=float, default=None, help="peak GPU memory on the serving VM, if measured (resources.csv)")
    p.add_argument("--note", default="", help="free-text note stored in metadata.json")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: write logs/ and .cache/ under this root instead of the repository")
    a = p.parse_args(argv)
    if not 1 <= a.prompts <= len(CHORES):
        p.error(f"--prompts must be between 1 and {len(CHORES)}")
    if a.parallel < 1 or a.timeout_sec < 1:
        p.error("--parallel and --timeout-sec must be positive")
    return a


def opencode_version() -> str:
    out = subprocess.run([str(OPENCODE_BIN), "--version"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or f"unavailable (exit {out.returncode})"


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if not OPENCODE_BIN.exists():
        raise SystemExit(f"opencode not found at {OPENCODE_BIN}")
    settings = asibench_opencode_settings()
    version = opencode_version()
    if settings["opencode_version"] and version != settings["opencode_version"]:
        raise SystemExit(f"opencode {version} on the host, the ASI-Bench lock pins {settings['opencode_version']}")
    baseline = load_baseline(a.baseline) if a.baseline else None
    run_id = a.run_id or new_run_id(*(["opencode_smoke", a.version] if a.version else ["opencode_smoke"]))
    root = Path(a.out_root).resolve() if a.out_root else REPO_ROOT
    chores = CHORES[: a.prompts]
    items = [(c, *materialize(c, a.seed)) for c in chores]
    prompt_sha = prompt_set_sha256(items)
    config = opencode_config(a.api_base, a.model_name, a.api_key, settings["tool_mode"])
    with RunDir(run_id, root=root, log_name="eval.log") as run:
        work_root = root / ".cache" / "opencode_smoke" / run_id
        run.log(f"run_id={run_id} model={a.model_name} api_base={a.api_base} chores={len(chores)} parallel={a.parallel} "
                f"timeout={a.timeout_sec}s opencode={version}")
        ep = preflight(run, a, docker=True)   # type: ignore[arg-type]  # uses only .error, .api_base, .model_name
        image = ensure_image(run)
        run.write_metadata(
            run_id=run_id, purpose="opencode process smoke (data plan WP10; section 1 guard); no scoring, no judge",
            version=a.version, model_name=a.model_name, endpoint=ep, model_revision=model_lock(),
            config={"parallel": a.parallel, "timeout_sec": a.timeout_sec, "n_prompts": len(chores),
                    "opencode": {"version": version, "binary": str(OPENCODE_BIN.resolve()), "binary_sha256": sha256_file(OPENCODE_BIN.resolve()),
                                 "command": opencode_args(a.model_name), "prompt_via": "stdin", "config_content": redacted(config),
                                 "env": {"HOME": "/home/agent (fresh per chore)", "OPENCODE_DISABLE_AUTOUPDATE": "1"},
                                 "matches": "ASI-Bench opencode_cli adapter (capture: " + str(ADAPTER_CAPTURE.relative_to(REPO_ROOT)) + ")"},
                    "asibench_lock": {"file": str(ASIBENCH_LOCK.relative_to(REPO_ROOT)), "sha256": sha256_file(ASIBENCH_LOCK), **settings},
                    "sandbox": {"image": image, "dockerfile": SANDBOX_DOCKERFILE, "limits": list(SANDBOX_LIMITS), "network": "host"}},
            data_version={"seed": a.seed, "prompt_set_sha256": prompt_sha, "is_reference_set": prompt_sha == REFERENCE_PROMPT_SET_SHA256,
                          "chores": [{"id": c.id, "title": c.title, "outputs": list(c.outputs)} for c in chores]},
            baseline=str(a.baseline) if a.baseline else None, code=git_info(), hardware=hardware_info(),
            seed=f"inputs generated with seed {a.seed}; opencode sends no sampling seed", command=command_line(), note=a.note,
            start_time=utc_now(), status="running")
        ctx = Context(run, run_id, work_root, image, a.model_name, json.dumps(config), a.timeout_sec, Sandboxes())
        pool = ThreadPoolExecutor(max_workers=a.parallel)
        try:
            futures = [pool.submit(run_chore, ctx, c, files, prompt) for c, files, prompt in items]
            rows = []
            for (c, _, _), fut in zip(items, futures):
                try:
                    rows.append(fut.result())
                except Exception as exc:  # noqa: BLE001 - recorded as an environment failure, never dropped
                    run.error(f"{c.id}: harness error: {type(exc).__name__}: {exc}")
                    rows.append(crashed_row(c, exc))
        finally:
            ctx.sandboxes.kill_all()
            pool.shutdown(wait=True, cancel_futures=True)
        for r in rows:
            run.metric(r)
        summary = {"run_id": run_id, "version": a.version, "model_name": a.model_name, **summarize(rows, prompt_sha)}
        summary["guard"] = guard(summary, baseline)
        write_json(run.run_dir / "summary.json", summary)
        run.log("per-prompt table\n" + table(rows))
        g = summary["guard"]
        run.log(f"totals: finished without user {summary['n_finished_without_user']}/{summary['n_prompts']}, files ok "
                f"{summary['n_files_ok']}/{summary['n_prompts']}, parse failures {summary['parse_failures']}/{summary['tool_calls']} "
                f"({summary['parse_failure_rate_pct']}%), turns {summary['turns_total']}, tokens in {summary['input_tokens']} out "
                f"{summary['output_tokens']} reasoning {summary['reasoning_tokens']}; guard {'PASS' if g['pass'] else 'FAIL'} {g['checks']}"
                + (f"; vs baseline {g['baseline']['run_id']}: {g['baseline']['parse_failures']}/{g['baseline']['tool_calls']}"
                   if g["baseline"] else ""))
        duration, rates = run.elapsed(), compute_rates()
        run.write_resources(resource_rows(duration, serving_gpus=a.serving_gpus, gpu_usd_per_hour=rates["gpu_vm_usd_per_hour"], peak_gpu_mem_gb=a.peak_gpu_mem_gb,
                                          orchestrator_cpus=os.cpu_count()))
        run.write_metadata(end_time=utc_now(), duration_sec=round(duration, 1), status="finished", summary=summary)
        run.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
