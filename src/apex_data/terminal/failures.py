"""Strong failure markers on a Terminus-2 terminal screen (v7): did the previous command really
fail? render/turn_roles.error_lines also matches a program's own error messages for bad input (`Error: file not found:
x`, printed on purpose while the agent tests error handling) and harmless broken pipes; hand-labelled on the 71 v7 tail
rows, only 43 (61%) of its hits were genuine failures. This detector keeps the strong markers only:

  traceback      `Traceback (most recent call last)`, unless every exception on the screen is a BrokenPipeError
  shell          `command not found`; `No such file or directory`, `Permission denied`, `cannot access`, `cannot open`
                 on a line that is not the program's own message (lines starting with error/fatal are skipped)
  tests          ^FAIL / ^FAILED, `--- FAIL`, `<n> failed`, AssertionError
  python         SyntaxError / IndentationError (python -c prints no traceback header)
  build/install  compiler `file:line[:col]: (fatal )error:`, pip `ERROR: Could not find` / `No matching distribution`,
                 `npm ERR!`, apt `E: Unable to locate`
  crashes        Segmentation fault, core dumped, `panicked`, ^panic:

Exit codes the agent echoes (`echo "Exit: $?"`) are not markers: agents print them mostly while testing error handling.
On those 71 rows this detector flags 44, 43 of them genuine (98%).
"""
from __future__ import annotations

import re

from apex_data.render.turn_roles import _ECHO_LINE, _PROMPT_LINE

TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
EXCEPTION = re.compile(r"^\s*([\w.]*(?:Error|Exception)):")      # `Name:`; not "Exception ignored in" (broken pipes)
SHELL_MISS = re.compile(r"No such file or directory|Permission denied|cannot access|cannot open")
OWN_MESSAGE = re.compile(r"^\s*(?:error|fatal|warning)\b", re.I)
STRONG = re.compile(
    r"command not found"
    r"|^\s*FAIL(?:ED)?\b|--- FAIL|\b[1-9]\d* failed\b|AssertionError"
    r"|^\s*(?:SyntaxError|IndentationError|TabError):"
    r"|^[^\s:]+:\d+(?::\d+)?: (?:fatal )?error:"
    r"|^ERROR: (?:Could not find a version|No matching distribution)|npm ERR!|^E: Unable to locate"
    r"|Segmentation fault|core dumped|\bpanicked\b|^panic:")


def failure_lines(output: str) -> list[str]:
    """Lines of a terminal output that show a genuine failure (echoed commands and heredoc continuations skipped)."""
    lines = [ln for ln in output.splitlines() if not _PROMPT_LINE.match(ln) and not _ECHO_LINE.match(ln)]
    out = [ln for ln in lines if STRONG.search(ln) or (SHELL_MISS.search(ln) and not OWN_MESSAGE.match(ln))]
    if any(TRACEBACK.search(ln) for ln in lines):
        excs = {m.group(1) for ln in lines if (m := EXCEPTION.match(ln))}
        if not excs or excs - {"BrokenPipeError"}:
            out += [ln for ln in lines if TRACEBACK.search(ln)]
    return out
