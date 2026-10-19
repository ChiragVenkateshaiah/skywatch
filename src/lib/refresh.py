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

from dataclasses import dataclass, field
from typing import Any

READY, READY_SLOW, NOT_READY = "READY", "READY_SLOW", "NOT_READY"


def step_timings(run: dict[str, Any]) -> list[dict[str, Any]]:
    """`run` = `w.jobs.get_run(run_id).as_dict()` (or the REST JSON). Tasks still running have
    no `end_time`; they are reported with `total_s = None`."""
    out = []
    for t in run.get("tasks", []):
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
