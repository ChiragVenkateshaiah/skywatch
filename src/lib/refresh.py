"""The pre-interview refresh's readiness verdict (docs/DEMO_REFRESH_PLAN.md) — pure Python, no
Spark/SDK, so it has a real pytest suite (tests/test_refresh.py). `demo_refresh_check.py` gathers
the inputs (the job run's JSON from the Jobs API, freshness queries) and calls these.

  step_timings  Jobs API `get_run` JSON -> one row per task: setup / execution / total seconds,
                result state — the record of where the 10-minute budget went
  readiness     the inputs -> READY / READY_SLOW / NOT_READY + the reasons

Ingestion freshness is judged on `gold_tracks` (what the poller + pipeline + gold actually
produced), not on `predictions`: at night a fresh burst can hold zero KATL inbounds, and
`score_eta.py` then writes no rows — that is a quiet sky, not a failed refresh.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

READY, READY_SLOW, NOT_READY = "READY", "READY_SLOW", "NOT_READY"


def latest_attempts(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The run JSON lists every attempt of a retried task under the same `task_key`; keep the
    latest one (highest `attempt_number`, then latest start) per task."""
    best: dict[str, dict[str, Any]] = {}
    for t in tasks:
        k = t["task_key"]
        rank = (t.get("attempt_number") or 0, t.get("start_time") or 0)
        if k not in best or rank > (best[k].get("attempt_number") or 0, best[k].get("start_time") or 0):
            best[k] = t
    return list(best.values())


def step_timings(run: dict[str, Any]) -> list[dict[str, Any]]:
    """`run` = `w.jobs.get_run(run_id).as_dict()` (or the REST JSON), one row per task (latest
    attempt). Tasks still running have no `end_time`; they are reported with `total_s = None`."""
    out = []
    for t in latest_attempts(run.get("tasks", [])):
        start, end = t.get("start_time"), t.get("end_time")
        out.append({
            "task": t["task_key"],
            "state": (t.get("state") or {}).get("result_state") or (t.get("state") or {}).get("life_cycle_state"),
            "setup_s": round((t.get("setup_duration") or 0) / 1000),
            "exec_s": round((t.get("execution_duration") or 0) / 1000),
            "total_s": round((end - start) / 1000) if start and end else None,
            "start_time": start,
        })
    return sorted(out, key=lambda r: r["start_time"] or 0)


@dataclass
class Readiness:
    status: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def readiness(
    *,
    steps: list[dict[str, Any]],
    gold_age_min: float | None,
    forecast_age_min: float | None,
    briefing_age_min: float | None,
    briefing_source: str | None,
    n_inbound: int,
    elapsed_min: float,
    max_age_min: float = 15,
    budget_min: float = 10,
    self_task: str = "check",
) -> Readiness:
    reasons, warnings = [], []

    failed = [s["task"] for s in steps if s["task"] != self_task and s["state"] not in ("SUCCESS", None)]
    if failed:
        reasons.append(f"failed steps: {', '.join(failed)}")

    for name, age in (("ingested data (gold_tracks)", gold_age_min),
                      ("demand forecast", forecast_age_min),
                      ("ops briefing", briefing_age_min)):
        if age is None:
            reasons.append(f"{name}: no rows")
        elif age > max_age_min:
            reasons.append(f"{name}: {age:.0f} min old (> {max_age_min:.0f})")

    if briefing_source == "template":
        warnings.append("briefing used the template fallback (LLM call failed or was rejected)")
    if n_inbound == 0:
        warnings.append("no inbound aircraft in the fresh burst — quiet sky, chat answers will be thin")
    slow = [f"{s['task']} setup {s['setup_s']} s" for s in steps if s["setup_s"] > 60]
    if slow:
        warnings.append("slow task start: " + ", ".join(slow))

    if reasons:
        return Readiness(NOT_READY, reasons, warnings)
    if elapsed_min > budget_min:
        warnings.insert(0, f"took {elapsed_min:.1f} min (budget {budget_min:.0f})")
        return Readiness(READY_SLOW, [], warnings)
    return Readiness(READY, [], warnings)


# --------------------------------------------------------------------- ops agent (DATA-2)
# Used by scripts/demo_ready.py. Only TRANSIENT failures are retried, once, by a Jobs repair
# run; everything else is reported with a diagnosis, because re-running a code or data
# failure just burns the minutes before an interview.
TRANSIENT, AUTH, CODE, DATA, UNKNOWN = "transient", "auth", "code", "data", "unknown"

_PATTERNS = [
    (AUTH, ("invalid_grant", "refresh token is invalid", "permission_denied", "unauthorized",
            "403", "does not have permission")),
    (TRANSIENT, ("internal_error", "transient", "retry the run", "temporarily unavailable",
                 "request_limit_exceeded", "too many requests", "429", "could not reach driver",
                 "connection reset", "service unavailable", "503")),
    (DATA, ("not ready: ", "no rows", "min old", "table or view not found", "table_or_view_not_found")),
    (CODE, ("traceback", "error:", "exception", "keyerror", "valueerror", "typeerror",
            "attributeerror", "nameerror", "assertionerror", "runtimeerror")),
]


def classify_failure(text: str | None) -> str:
    """First matching class wins, in the order above: auth beats transient (a 403 is not worth a
    retry), transient beats code (a platform error often arrives wrapped in a Python traceback)."""
    t = (text or "").lower()
    for kind, needles in _PATTERNS:
        if any(n in t for n in needles):
            return kind
    return UNKNOWN


def progress_events(prev: dict[str, str], steps: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Task state changes since the last poll, as (task, new_state) in run order."""
    return [(s["task"], str(s["state"])) for s in steps if prev.get(s["task"]) != str(s["state"])]


def failed_tasks(steps: list[dict[str, Any]], self_task: str = "check") -> list[str]:
    """Tasks that ended in a non-success state. `UPSTREAM_FAILED` / `EXCLUDED` are skipped:
    they only say an earlier task failed, so the earlier one is the one worth diagnosing."""
    bad = {"FAILED", "TIMEDOUT", "INTERNAL_ERROR", "CANCELED"}
    return [s["task"] for s in steps if s["state"] in bad and s["task"] != self_task]


_ANSI = re.compile(r"\x1b\[[0-9;]*m|\[[0-9;]*m")
_QUOTE = re.compile(r'"([^"\n]{12,})"|`([^`\n]{12,})`')


def strip_ansi(text: str) -> str:
    """Notebook traces carry terminal colour codes; they waste tokens and break substring checks."""
    return _ANSI.sub("", text or "")


def quote_is_grounded(answer: str, evidence: str) -> bool:
    """The diagnosis must quote at least one error line (>= 12 chars, in "..." or `...`) that
    really appears in the evidence. Attempt 1 of the agent invented a FileNotFoundException that
    was nowhere in the run's output — this is the check that catches that."""
    norm = " ".join(strip_ansi(evidence).split())
    quotes = [q1 or q2 for q1, q2 in _QUOTE.findall(answer or "")]
    return any(" ".join(q.split()) in norm for q in quotes)
