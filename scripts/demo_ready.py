"""Get SkyWatch demo-ready — the ops agent for the pre-interview refresh (roadmap DATA-2a,
docs/DEMO_REFRESH_PLAN.md). Run it locally a few minutes before an interview:

    python scripts/demo_ready.py                 # start skywatch_demo_refresh, watch, report
    python scripts/demo_ready.py --report-only   # just report the latest refresh

What it does:
  1. starts `skywatch_demo_refresh` and prints every task state change as it happens
  2. reads the job's own verdict (`refresh_audit`: READY / READY_SLOW / NOT_READY)
  3. a failed task is classified (lib.refresh.classify_failure). Only a TRANSIENT failure is
     retried — once, as a Jobs *repair* run, which re-runs the failed task and the ones after it
     but not the preflight/reset. Auth / code / data failures are never retried.
  4. still not ready -> an LLM (Foundation Model API) gets the failed tasks' real errors and
     writes a short diagnosis + one fix, which must quote the real error line or the raw error is
     shown instead. The retry decision is code, never the model's.
  5. writes the run as one MLflow trace (roadmap OPT-2): run > task > setup / execution, at the
     run's real timestamps, into the bundle's `skywatch-traces` experiment — no compute. Also
     with --report-only, so a refresh started from the App gets traced too; a run is traced once.
     Needs `mlflow-tracing` (requirements-dev.txt); without it this step is skipped.

Exit code 0 = READY or READY_SLOW, 1 = NOT_READY.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbx  # noqa: E402
from lib import tracing  # noqa: E402
from lib.refresh import (  # noqa: E402
    TRANSIENT, classify_failure, failed_tasks, latest_attempts, progress_events, quote_is_grounded,
    step_timings, strip_ansi, trace_spans,
)

JOB_KEY = "skywatch_demo_refresh"
DONE = {"TERMINATED", "INTERNAL_ERROR", "SKIPPED"}
QUIET = {"BLOCKED", "PENDING", "QUEUED"}  # waiting states — not worth a line each


def bundle_ids(profile: str, target: str) -> tuple[int, str, str | None]:
    out = subprocess.check_output(
        ["databricks", "bundle", "summary", "-t", target, "--profile", profile, "-o", "json"],
        cwd=ROOT, stderr=subprocess.DEVNULL,
    )
    s = json.loads(out)
    job_id = s["resources"]["jobs"][JOB_KEY]["id"]
    wh = s.get("variables", {}).get("warehouse_id", {}).get("value")
    exp = s["resources"].get("experiments", {}).get("skywatch_traces", {}).get("id")
    return int(job_id), wh, exp


def trace_run(profile: str, experiment: str | None, run: dict) -> str:
    """One trace per finished run (OPT-2); skipped if this run already has one."""
    try:
        import mlflow
    except ImportError:
        return "not traced (pip install -r requirements-dev.txt for mlflow-tracing)"
    if not run.get("end_time"):
        return "not traced (run still going)"
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    if not tracing.setup(experiment):
        return "not traced (no skywatch-traces experiment — bundle deploy creates it)"
    try:
        if len(mlflow.search_traces(filter_string=f"tags.run_id = '{run['run_id']}'", max_results=1)):
            return "already traced"
        tid = tracing.write_plan(trace_spans(run), {"kind": "refresh", "run_id": str(run["run_id"])})
        mlflow.flush_trace_async_logging()
        return f"traced as {tid}"
    except Exception as exc:  # noqa: BLE001 — the report matters more than its trace
        return f"not traced ({type(exc).__name__}: {str(exc)[:120]})"


def elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 60:02d}:{s % 60:02d}"


def watch(ws: dbx.Workspace, run_id: int, t0: float, poll_s: int = 10) -> dict:
    seen: dict[str, str] = {}
    while True:
        run = ws.get("/api/2.2/jobs/runs/get", {"run_id": run_id})
        steps = step_timings(run)
        for task, state in progress_events(seen, steps):
            seen[task] = state
            if state in QUIET:
                continue
            s = next(x for x in steps if x["task"] == task)
            extra = f"  (setup {s['setup_s']} s, exec {s['exec_s']} s)" if s["total_s"] else ""
            print(f"[{elapsed(t0)}] {task:<22} {state}{extra}", flush=True)
        if run["state"]["life_cycle_state"] in DONE:
            return run
        time.sleep(poll_s)


def task_errors(ws: dbx.Workspace, run: dict, tasks: list[str]) -> dict[str, str]:
    """Error text for each failed task: the error, the end of its trace and of its logs."""
    out = {}
    by_key = {t["task_key"]: t for t in latest_attempts(run.get("tasks", []))}
    for key in tasks:
        try:
            o = ws.get("/api/2.2/jobs/runs/get-output", {"run_id": by_key[key]["run_id"]})
        except urllib.error.HTTPError as exc:
            out[key] = f"(could not read output: HTTP {exc.code})"
            continue
        parts = [o.get("error") or "", (o.get("error_trace") or "")[-1500:], (o.get("logs") or "")[-800:]]
        out[key] = strip_ansi("\n".join(p for p in parts if p)).strip() or (by_key[key].get("state") or {}).get("state_message", "")
    return out


def latest_audit(ws: dbx.Workspace, wh: str, run_id: int) -> dict | None:
    df = ws.sql(wh, f"""SELECT checked_at, status, reasons, warnings, data_as_of, elapsed_min, n_inbound,
                               n_flags, briefing_source FROM refresh_audit
                        WHERE job_run_id = {run_id} ORDER BY checked_at DESC LIMIT 1""")
    return None if df.empty else df.iloc[0].to_dict()


# ---------------------------------------------------------------------------- the agent
# First version gave the model tools (list_tasks / get_task_error / get_refresh_audit) to fetch
# the evidence itself. Tested on a real past failure (run 797709381635769, "ValueError: Input
# contains NaN"), Maverick wrote a fake tool call as plain text and invented a
# FileNotFoundException. So: the evidence goes straight into the prompt, and the answer must quote
# an error line that really is in it (lib.refresh.quote_is_grounded) — else the raw error is shown.
AGENT_SYSTEM = """You diagnose a failed run of SkyWatch's pre-interview refresh job on Databricks.
The job runs, in order: prepare (preflight + reset), poll, medallion, gold, score_eta, score_demand,
score_irregularities, brief, check. The user message holds the run's real evidence.
Answer in under 120 words, plain text, in three short parts:
1. Which task failed first.
2. The likely cause, quoting the exact error line from the evidence in double quotes.
3. One concrete next step.
Use only the evidence given. Never invent an error message. If the evidence is not enough, say so."""


def diagnose(ws: dbx.Workspace, endpoint: str, steps: list[dict], errors: dict[str, str],
             audit: dict | None) -> tuple[str, bool]:
    """(text, grounded). `grounded = False` means the answer didn't quote the real evidence."""
    tasks = "\n".join(f"- {s['task']}: {s['state']} (setup {s['setup_s']} s, exec {s['exec_s']} s)" for s in steps)
    errs = "\n\n".join(f"### {t}\n{e[-2500:]}" for t, e in errors.items()) or "(no task error text recorded)"
    verdict = json.dumps({k: str(v) for k, v in (audit or {}).items()}) if audit else "(the check did not run)"
    evidence = f"Tasks:\n{tasks}\n\nErrors:\n{errs}\n\nReadiness verdict:\n{verdict}"
    r = ws.post(f"/serving-endpoints/{endpoint}/invocations", {
        "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": evidence}],
        "temperature": 0, "max_tokens": 400,
    })
    text = (r["choices"][0]["message"].get("content") or "").strip()
    return text, quote_is_grounded(text, errs + verdict)


# ----------------------------------------------------------------------------- warm-up
GENIE_SPACE = "01f1a7adfd07143d8c92fd72fa8fa48f"
CHAT_ENDPOINTS = ["databricks-llama-4-maverick", "databricks-meta-llama-3-3-70b-instruct"]
TOOL_ARGS = {"next_arrivals": "(1)", "aircraft": "('X')", "demand_outlook": "(90.0)", "model_status": "('eta_touchdown')"}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def warm_up(ws: dbx.Workspace, wh: str) -> None:
    """Touch everything the demo's first questions will touch, so no cold start lands on camera:
    the first managed-MCP call took 37 s cold, a cold Genie query ~16 s (docs/CHAT_PLAN.md §2).
    The App's own caches (tool list, Model 1 for explain_eta) warm on the first question asked
    in the App — the demo script does that before recording."""
    def step(name, fn):
        t0 = time.time()
        try:
            detail = fn()
            print(f"  warm-up ok    {name:<34} {detail}  {time.time() - t0:.1f} s", flush=True)
        except Exception as exc:  # noqa: BLE001 — a warm-up failure is a warning, the refresh verdict stands
            print(f"  warm-up FAIL  {name:<34} {type(exc).__name__}: {str(exc)[:120]}", flush=True)

    tools = ws.sql(wh, "SELECT routine_name FROM skywatch.information_schema.routines "
                       "WHERE routine_schema = 'chat' AND routine_type = 'FUNCTION'")["routine_name"].tolist()
    for t in sorted(tools):
        step(f"tool {t}", lambda t=t: f"{len(ws.sql(wh, f'SELECT * FROM skywatch.chat.{t}{TOOL_ARGS.get(t, chr(40) + chr(41))}'))} rows")

    def mcp(path, body):
        import urllib.request
        req = urllib.request.Request(ws.host + path, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {ws.token()}", "Content-Type": "application/json", **MCP_HEADERS})
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    step("MCP functions tools/list", lambda: f"{len(mcp('/api/2.0/mcp/functions/skywatch/chat', {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})['result']['tools'])} tools")
    step("MCP Genie query", lambda: json.loads(mcp(f"/api/2.0/mcp/genie/{GENIE_SPACE}", {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": f"query_space_{GENIE_SPACE}", "arguments": {"query": "How many tables do you have?"}}})
        ["result"]["content"][0]["text"]).get("status"))
    for ep in CHAT_ENDPOINTS:
        step(f"endpoint {ep}", lambda ep=ep: ws.post(f"/serving-endpoints/{ep}/invocations", {
            "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 5})["choices"][0]["message"]["content"][:10])


# ------------------------------------------------------------------------------ report
def report(audit: dict | None, run: dict, retried: bool) -> str:
    steps = step_timings(run)
    lines = ["", "=" * 64]
    if audit:
        lines.append(f"  {audit['status']}  —  {float(audit['elapsed_min']):.1f} min"
                     + ("  (after one transient retry)" if retried else ""))
        if audit.get("data_as_of") is not None:
            lines.append(f"  data as of {audit['data_as_of']:%Y-%m-%d %H:%M} UTC · "
                         f"{audit['n_inbound']} inbound · {audit['n_flags']} flagged · "
                         f"briefing: {audit['briefing_source']}")
        for label, key in (("reason", "reasons"), ("warning", "warnings")):
            for x in (audit.get(key) or "").split(" | "):
                if x and x != "None":
                    lines.append(f"  {label}: {x}")
    else:
        lines.append("  NOT_READY  —  the readiness check did not write a verdict")
    slow = max((s for s in steps if s["total_s"]), key=lambda s: s["total_s"], default=None)
    if slow:
        lines.append(f"  slowest task: {slow['task']} ({slow['total_s']} s)")
    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="skywatch")
    ap.add_argument("--target", default="dev")
    ap.add_argument("--job-id", type=int)
    ap.add_argument("--warehouse-id")
    ap.add_argument("--endpoint", default="databricks-llama-4-maverick")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="report the latest refresh, start nothing")
    ap.add_argument("--no-warmup", action="store_true", help="skip the warm-up after a READY verdict")
    ap.add_argument("--trace-experiment", help="MLflow experiment id or path (default: the bundle's skywatch-traces)")
    args = ap.parse_args()

    ws = dbx.Workspace(args.profile)
    job_id, wh, exp = args.job_id, args.warehouse_id, args.trace_experiment
    if not (job_id and wh and exp):
        b_job, b_wh, b_exp = bundle_ids(args.profile, args.target)
        job_id, wh, exp = job_id or b_job, wh or b_wh, exp or b_exp
    t0 = time.time()

    if args.report_only:
        runs = ws.get("/api/2.2/jobs/runs/list", {"job_id": job_id, "limit": 1}).get("runs") or []
        if not runs:
            print("no refresh run found")
            return 1
        run = ws.get("/api/2.2/jobs/runs/get", {"run_id": runs[0]["run_id"]})
    else:
        run_id = ws.post("/api/2.2/jobs/run-now", {"job_id": job_id})["run_id"]
        print(f"started {JOB_KEY} run {run_id} — {ws.host}/jobs/{job_id}/runs/{run_id}", flush=True)
        run = watch(ws, run_id, t0)
    run_id, retried = run["run_id"], False

    failed = failed_tasks(step_timings(run))
    errors = task_errors(ws, run, failed) if failed else {}
    kinds = {t: classify_failure(e) for t, e in errors.items()}
    for t, k in kinds.items():
        print(f"  {t}: {k} failure")

    if failed and not args.no_retry and not args.report_only and all(k == TRANSIENT for k in kinds.values()):
        print(f"[{elapsed(t0)}] transient failure — repairing once (re-runs {', '.join(failed)} and later tasks)")
        latest_repair = max((r.get("repair_id") for r in run.get("repair_history", []) if r.get("repair_id")), default=None)
        body = {"run_id": run_id, "rerun_all_failed_tasks": True}
        if latest_repair:
            body["latest_repair_id"] = latest_repair
        ws.post("/api/2.2/jobs/runs/repair", body)
        retried = True
        time.sleep(5)
        run = watch(ws, run_id, t0)
        failed = failed_tasks(step_timings(run))
        errors = task_errors(ws, run, failed) if failed else {}

    audit = latest_audit(ws, wh, run_id) if wh else None
    print(report(audit, run, retried))
    print(f"  trace: {trace_run(args.profile, exp, run)}")

    status = audit["status"] if audit else "NOT_READY"
    if status != "NOT_READY" and not args.no_warmup and wh:
        print("\nWarm-up (so no cold start lands on camera):")
        warm_up(ws, wh)
    if status == "NOT_READY":
        print("\nDiagnosis (LLM agent, from the run's own errors):")
        try:
            text, grounded = diagnose(ws, args.endpoint, step_timings(run), errors, audit)
        except Exception as exc:  # noqa: BLE001 — the raw errors are still worth showing
            text, grounded = f"(agent unavailable: {type(exc).__name__}: {exc})", False
        print(text)
        if not grounded:
            print("\n  (the diagnosis above does not quote the run's real error — raw evidence below)")
            for t, e in errors.items():
                print(f"--- {t}\n{e[-800:]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
