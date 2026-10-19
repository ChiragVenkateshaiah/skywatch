"""Tests for src/lib/chat.py — the chat turn loop and its guards (docs/CHAT_PLAN.md §3). The LLM
is a scripted fake, the tools are fakes; nothing touches the network.
"""

from __future__ import annotations

import json

from lib.chat import (
    APP_TOOLS, ToolCall, ToolSpec, compact, data_as_of, looks_like_text_tool_call, run_turn,
    uc_call_sql, uc_tool_specs,
)

ROUTINES = [{"routine_name": "next_arrivals", "comment": "The next n aircraft."},
            {"routine_name": "arrival_picture", "comment": "Inbound counts."}]
PARAMS = [{"specific_name": "next_arrivals", "parameter_name": "n", "data_type": "INT", "ordinal_position": 0}]
TOOLS = uc_tool_specs(ROUTINES, PARAMS) + APP_TOOLS

PICTURE = compact(["inbound", "within_10_min", "scored_at"], [["113", "15", "2026-10-19T15:01:20.067Z"]])
NEXT = compact(["rank", "callsign", "eta_min", "touchdown_utc"], [["1", "DAL3100", "1.3", "14:57"]])


def _tool_msg(*calls):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def _text_msg(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


class FakeLLM:
    def __init__(self, *responses):
        self.responses, self.seen = list(responses), []

    def __call__(self, messages, schemas):
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


def _execute(name, args):
    return {"arrival_picture": PICTURE, "next_arrivals": NEXT}[name]


def _run(*responses, **kw):
    llm = FakeLLM(*responses)
    return run_turn("How busy is KATL?", llm=llm, execute=_execute, tools=TOOLS, **kw), llm


# ------------------------------------------------------------------------------- tools
class TestToolSpecs:
    def test_specs_from_information_schema(self):
        spec = {t.name: t for t in TOOLS}["next_arrivals"]
        assert spec.params == [("n", "INT")] and spec.kind == "uc"
        assert spec.schema()["function"]["parameters"] == {
            "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

    def test_no_param_function(self):
        assert {t.name: t for t in TOOLS}["arrival_picture"].params == []

    def test_sql_literals_are_typed(self):
        spec = ToolSpec("demand_outlook", "", [("aar_per_hour", "DOUBLE")], "uc")
        assert uc_call_sql(spec, {"aar_per_hour": "90"}) == "SELECT * FROM skywatch.chat.demand_outlook(90.0)"

    def test_sql_injection_is_stripped(self):
        spec = ToolSpec("aircraft", "", [("cs", "STRING")], "uc")
        sql = uc_call_sql(spec, {"cs": "DAL1'); DROP TABLE x; --"})
        assert sql == "SELECT * FROM skywatch.chat.aircraft('DAL1 DROP TABLE x --')"

    def test_compact_truncates(self):
        r = compact(["a"], [[i] for i in range(30)])
        assert r["row_count"] == 30 and len(r["rows"]) == 25 and r["truncated"]


# -------------------------------------------------------------------------------- turn
class TestTurn:
    def test_tool_then_grounded_answer(self):
        turn, _ = _run(_tool_msg(("arrival_picture", {})),
                       _text_msg("113 aircraft are inbound; 15 are predicted to land within 10 minutes."))
        assert turn.source == "llm" and turn.check.passed and turn.rounds == 2
        assert [c.name for c in turn.calls] == ["arrival_picture"]
        assert turn.prompt_tokens == 220 and turn.completion_tokens == 40

    def test_values_from_iso_timestamp_are_allowed(self):
        turn, _ = _run(_tool_msg(("arrival_picture", {})), _text_msg("As of 15:01 UTC, 113 are inbound."))
        assert turn.check.passed, turn.check.reasons

    def test_invented_number_gets_one_repair(self):
        turn, llm = _run(_tool_msg(("arrival_picture", {})),
                         _text_msg("120 aircraft are inbound."),
                         _text_msg("113 aircraft are inbound."))
        assert turn.repaired and turn.source == "llm" and turn.answer == "113 aircraft are inbound."
        assert "number 120 not in tool results" in llm.seen[-1][-1]["content"]

    def test_second_failed_check_falls_back(self):
        turn, _ = _run(_tool_msg(("arrival_picture", {})), _text_msg("120 inbound."), _text_msg("121 inbound."))
        assert turn.source == "fallback" and not turn.check.passed
        assert "arrival_picture (1 rows)" in turn.answer

    def test_text_tool_call_is_nudged_not_shown(self):
        turn, llm = _run(_text_msg('next_arrivals(n=3)'), _tool_msg(("next_arrivals", {"n": 3})),
                         _text_msg("Next is DAL3100, landing in 1.3 minutes at 14:57."))
        assert turn.nudged and turn.source == "llm" and "DAL3100" in turn.answer
        assert "tool-calling interface" in llm.seen[1][-1]["content"]

    def test_repeated_text_tool_call_falls_back(self):
        turn, _ = _run(_text_msg('<|python_tag|>{"name": "next_arrivals"}'), _text_msg("next_arrivals(3)"))
        assert turn.source == "fallback" and turn.check.reasons == ["no checked answer within the round limit"]

    def test_refusal_without_tools_passes(self):
        turn, _ = _run(_text_msg("I can't help with the weather in Paris. I can answer questions about "
                                 "KATL arrivals, demand and the SkyWatch models."))
        assert turn.source == "llm" and turn.calls == [] and turn.rounds == 1

    def test_unknown_tool_is_an_error_the_model_sees(self):
        turn, llm = _run(_tool_msg(("book_flight", {})), _text_msg("I can't book flights."))
        assert turn.calls[0].error == "unknown tool book_flight"
        assert json.loads(llm.seen[1][-1]["content"]) == {"error": "unknown tool book_flight"}

    def test_tool_exception_is_reported_to_the_model(self):
        def boom(name, args):
            raise RuntimeError("warehouse asleep")
        turn = run_turn("q", llm=FakeLLM(_tool_msg(("arrival_picture", {})), _text_msg("The data is unavailable.")),
                        execute=boom, tools=TOOLS)
        assert turn.calls[0].error == "RuntimeError: warehouse asleep" and turn.source == "llm"

    def test_round_limit(self):
        turn, _ = _run(*[_tool_msg(("arrival_picture", {}))] * 4)
        assert turn.rounds == 4 and turn.source == "fallback"

    def test_history_answers_stay_allowed(self):
        first, _ = _run(_tool_msg(("arrival_picture", {})), _text_msg("113 aircraft are inbound."))
        second = run_turn("And how many was that?", llm=FakeLLM(_text_msg("It was 113.")),
                          execute=_execute, tools=TOOLS, history=[first])
        assert second.check.passed and second.calls == []

    def test_history_is_sent_as_plain_turns(self):
        first, _ = _run(_tool_msg(("arrival_picture", {})), _text_msg("113 aircraft are inbound."))
        llm = FakeLLM(_text_msg("Yes."))
        run_turn("Really?", llm=llm, execute=_execute, tools=TOOLS, history=[first])
        assert [m["role"] for m in llm.seen[0]] == ["system", "user", "assistant", "user"]


class TestHelpers:
    def test_text_tool_call_detection(self):
        names = ["next_arrivals", "aircraft"]
        assert looks_like_text_tool_call("aircraft(cs='DAL1')", names)
        assert looks_like_text_tool_call('{"name": "next_arrivals", "parameters": {}}', names)
        assert not looks_like_text_tool_call("The next aircraft lands in 2 minutes.", names)

    def test_data_as_of_is_the_oldest_stamp(self):
        calls = [ToolCall("a", {}, compact(["scored_at"], [["2026-10-19T15:01:20Z"]])),
                 ToolCall("b", {}, compact(["generated_at"], [["2026-10-19T14:58:00Z"]])),
                 ToolCall("c", {}, error="x")]
        assert data_as_of(calls) == "2026-10-19T14:58:00Z"
        assert data_as_of([]) is None


def test_genie_ask_polls_until_completed():
    from lib.chat import genie_ask
    replies = [
        {"status": "ASKING_AI", "conversationId": "c1", "messageId": "m1", "content": {}},
        {"status": "COMPLETED", "conversationId": "c1", "messageId": "m1", "content": {
            "textAttachments": ["Hour 12 is busiest."],
            "queryAttachments": [{"query": "SELECT 12", "statement_response": {
                "manifest": {"schema": {"columns": [{"name": "hour"}, {"name": "avg"}]}},
                "result": {"data_array": [["12", "76.04"]]}}}]}},
    ]
    sent = []

    def post(path, body):
        sent.append((path, body["params"]["name"], body["params"]["arguments"]))
        return {"result": {"content": [{"text": json.dumps(replies.pop(0))}]}}

    out = genie_ask(post, "S1", "busiest hour?", sleep=lambda s: None)
    assert out == {"answer": "Hour 12 is busiest.", "sql": "SELECT 12", "columns": ["hour", "avg"],
                   "rows": [["12", "76.04"]], "row_count": 1}
    assert [n for _, n, _ in sent] == ["query_space_S1", "poll_response_S1"]
    assert sent[1][2] == {"conversation_id": "c1", "message_id": "m1"}


DEMAND = compact(["bin_start_utc", "predicted", "arrivals_seen_by_skywatch_today"], [["15:00", "12.7", "6"]])


DTOOLS = TOOLS + [ToolSpec("demand_outlook", "Demand forecast.", [("aar_per_hour", "DOUBLE")], "uc")]


def _exec_demand(name, args):
    return DEMAND


class TestMeaningRules:
    def test_landed_today_without_skywatch_is_repaired(self):
        llm = FakeLLM(_tool_msg(("demand_outlook", {})), _text_msg("6 flights have landed at Atlanta today."),
                      _text_msg("SkyWatch has seen 6 landings today; that is not the airport's full count."))
        turn = run_turn("How many landed today?", llm=llm, execute=_exec_demand, tools=DTOOLS)
        assert turn.repaired and turn.source == "llm" and turn.answer.startswith("SkyWatch has seen 6")
        assert any(r.startswith("meaning:") for r in turn.repair_reasons)

    def test_correct_phrasing_passes_first_time(self):
        llm = FakeLLM(_tool_msg(("demand_outlook", {})), _text_msg("SkyWatch has seen 6 landings today."))
        turn = run_turn("How many landed today?", llm=llm, execute=_exec_demand, tools=DTOOLS)
        assert not turn.repaired and turn.source == "llm"

    def test_unrelated_today_sentence_is_not_flagged(self):
        llm = FakeLLM(_tool_msg(("demand_outlook", {})), _text_msg("The forecast peaks at 15:00 with 12.7 today."))
        assert run_turn("q", llm=llm, execute=_exec_demand, tools=DTOOLS).source == "llm"
