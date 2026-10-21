"""Tests for src/lib/refresh.py — the pre-interview refresh's readiness verdict
(docs/DEMO_REFRESH_PLAN.md). Pure Python; the notebook (demo_refresh_check.py) gathers the inputs.
"""

from __future__ import annotations

from lib.refresh import NOT_READY, READY, READY_SLOW, readiness, step_timings, trace_spans

T0 = 1_792_307_000_000


def _task(key, start_s, total_s, setup_s=4, state="SUCCESS"):
    return {
        "task_key": key,
        "start_time": T0 + start_s * 1000,
        "end_time": T0 + (start_s + total_s) * 1000 if total_s is not None else None,
        "setup_duration": setup_s * 1000,
        "execution_duration": (total_s - setup_s) * 1000 if total_s is not None else 0,
        "state": {"result_state": state} if state else {"life_cycle_state": "RUNNING"},
    }


RUN = {"tasks": [_task("poll", 30, 190), _task("prepare", 0, 30), _task("check", 600, None, state=None)]}


class TestStepTimings:
    def test_sorted_by_start_with_seconds(self):
        steps = step_timings(RUN)
        assert [s["task"] for s in steps] == ["prepare", "poll", "check"]
        assert steps[1] == {"task": "poll", "state": "SUCCESS", "setup_s": 4, "exec_s": 186,
                            "total_s": 190, "start_time": T0 + 30_000}

    def test_running_task_has_no_total(self):
        check = step_timings(RUN)[-1]
        assert check["total_s"] is None and check["state"] == "RUNNING"

    def test_empty_run(self):
        assert step_timings({}) == []


def _ok(**kw):
    args = dict(steps=step_timings(RUN), gold_age_min=3, forecast_age_min=2, briefing_age_min=1,
                briefing_source="llm", n_inbound=24, elapsed_min=8.7)
    args.update(kw)
    return readiness(**args)


class TestReadiness:
    def test_all_fresh_within_budget_is_ready(self):
        r = _ok()
        assert r.status == READY and r.reasons == [] and r.warnings == []

    def test_own_running_task_is_not_a_failure(self):
        assert _ok().status == READY  # "check" is RUNNING while it computes this

    def test_failed_step_is_not_ready(self):
        steps = step_timings({"tasks": [_task("gold", 0, 60, state="FAILED")]})
        r = _ok(steps=steps)
        assert r.status == NOT_READY and r.reasons == ["failed steps: gold"]

    def test_stale_or_missing_source_is_not_ready(self):
        r = _ok(gold_age_min=42, briefing_age_min=None)
        assert r.status == NOT_READY
        assert r.reasons == ["ingested data (gold_tracks): 42 min old (> 15)", "ops briefing: no rows"]

    def test_over_budget_is_ready_slow(self):
        r = _ok(elapsed_min=12.4)
        assert r.status == READY_SLOW and r.warnings[0] == "took 12.4 min (budget 10)"

    def test_quiet_sky_is_a_warning_not_a_failure(self):
        r = _ok(n_inbound=0)
        assert r.status == READY and "quiet sky" in r.warnings[0]

    def test_template_briefing_is_a_warning(self):
        assert "template fallback" in _ok(briefing_source="template").warnings[0]

    def test_slow_task_start_is_reported(self):
        steps = step_timings({"tasks": [_task("score_eta", 0, 770, setup_s=705)]})
        assert _ok(steps=steps).warnings == ["slow task start: score_eta setup 705 s"]


# --------------------------------------------------------------------- ops agent (DATA-2)
from lib.refresh import (  # noqa: E402
    AUTH, CODE, DATA, TRANSIENT, UNKNOWN, classify_failure, failed_tasks, progress_events,
)


class TestClassifyFailure:
    def test_platform_blip_is_transient(self):
        assert classify_failure("The service encountered a transient issue. Retry the run") == TRANSIENT

    def test_qps_limit_is_transient(self):
        assert classify_failure("REQUEST_LIMIT_EXCEEDED: Exceeded workspace QPS rate limit") == TRANSIENT

    def test_expired_login_is_auth_not_transient(self):
        assert classify_failure('{"error":"invalid_grant","error_description":"Refresh token is invalid"}') == AUTH

    def test_python_error_is_code(self):
        assert classify_failure("Traceback (most recent call last):\n KeyError: 'scored_at'") == CODE

    def test_transient_wrapped_in_traceback_is_still_transient(self):
        assert classify_failure("Traceback ...\nDatabricksError: INTERNAL_ERROR") == TRANSIENT

    def test_readiness_failure_is_data(self):
        assert classify_failure("RuntimeError: NOT READY: demand forecast: 42 min old (> 15)") == DATA

    def test_empty_is_unknown(self):
        assert classify_failure(None) == UNKNOWN and classify_failure("") == UNKNOWN


class TestProgress:
    def test_only_changes_are_reported(self):
        steps = [{"task": "prepare", "state": "SUCCESS"}, {"task": "poll", "state": "RUNNING"}]
        assert progress_events({"prepare": "SUCCESS"}, steps) == [("poll", "RUNNING")]

    def test_failed_tasks_skip_upstream_failed_and_self(self):
        steps = [{"task": "gold", "state": "FAILED"}, {"task": "score_eta", "state": "UPSTREAM_FAILED"},
                 {"task": "check", "state": "FAILED"}]
        assert failed_tasks(steps) == ["gold"]


from lib.refresh import latest_attempts, quote_is_grounded, strip_ansi  # noqa: E402


class TestAttemptsAndGrounding:
    def test_latest_attempt_wins(self):
        tasks = [{"task_key": "train", "attempt_number": 0, "start_time": 1, "run_id": 10},
                 {"task_key": "train", "attempt_number": 1, "start_time": 2, "run_id": 11},
                 {"task_key": "gold", "start_time": 0, "run_id": 12}]
        assert sorted(t["run_id"] for t in latest_attempts(tasks)) == [11, 12]

    def test_step_timings_has_one_row_per_task(self):
        run = {"tasks": [_task("train", 0, 60, state="FAILED") | {"attempt_number": 0},
                         _task("train", 70, 60, state="FAILED") | {"attempt_number": 1}]}
        assert [s["task"] for s in step_timings(run)] == ["train"]

    def test_strip_ansi(self):
        assert strip_ansi("\x1b[38;5;241mValueError\x1b[0m: x [43mm") == "ValueError: x m"

    EVIDENCE = "Traceback ...\nValueError: Input contains NaN.\n[Trace ID: 00-45bf]"

    def test_real_quote_is_grounded(self):
        assert quote_is_grounded('Cause: "ValueError: Input contains NaN." in by_band', self.EVIDENCE)

    def test_invented_quote_is_not_grounded(self):
        assert not quote_is_grounded('"java.io.FileNotFoundException: /dbfs/mnt/data"', self.EVIDENCE)

    def test_no_quote_is_not_grounded(self):
        assert not quote_is_grounded("The training failed because of NaNs.", self.EVIDENCE)


class TestTraceSpans:
    def test_run_task_and_phases_with_real_times(self):
        run = {"run_id": 7, "start_time": T0, "end_time": T0 + 700_000, "state": {"result_state": "SUCCESS"},
               "tasks": RUN["tasks"] + [{"task_key": "brief", "state": {"life_cycle_state": "SKIPPED"}}]}
        spans = trace_spans(run)
        assert [s["name"] for s in spans] == ["refresh run 7", "task prepare", "setup", "execution",
                                             "task poll", "setup", "execution", "task check", "setup"]
        root, poll = spans[0], spans[4]
        assert (root["start_ms"], root["end_ms"], root["attrs"]["result_state"]) == (T0, T0 + 700_000, "SUCCESS")
        assert (poll["parent"], poll["start_ms"], poll["end_ms"]) == (0, T0 + 30_000, T0 + 220_000)
        assert poll["attrs"] == {"state": "SUCCESS", "attempt": 0, "setup_s": 4, "exec_s": 186}
        setup, execution = spans[5], spans[6]
        assert setup["parent"] == execution["parent"] == 4
        assert (setup["end_ms"], execution["start_ms"], execution["end_ms"]) == (T0 + 34_000, T0 + 34_000, T0 + 220_000)
        # a running task (no end_time, no execution yet) ends with the run and has no execution span
        assert spans[7]["end_ms"] == T0 + 700_000 and spans[8]["name"] == "setup"

    def test_run_without_times_uses_its_tasks(self):
        spans = trace_spans({"run_id": 1, "tasks": [_task("prepare", 0, 30)]})
        assert (spans[0]["start_ms"], spans[0]["end_ms"]) == (T0, T0 + 30_000)
