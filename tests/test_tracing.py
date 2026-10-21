"""Tests for src/lib/tracing.py — the MLflow adapters of roadmap OPT-2. MLflow is replaced by a
fake module, so nothing touches the network and mlflow-tracing need not be installed."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest

from lib import tracing
from lib.chat import no_span


class FakeSpan:
    def __init__(self, name, **kw):
        self.name, self.kw, self.attrs, self.end_ns = name, kw, {}, None
        self.trace_id = "tr-fake"

    def set_attributes(self, a):
        self.attrs.update(a)

    def end(self, end_time_ns=None):
        self.end_ns = end_time_ns


@pytest.fixture
def fake_mlflow(monkeypatch):
    m = types.ModuleType("mlflow")
    m.spans, m.tags = [], {}

    @contextmanager
    def start_span(name, span_type):
        sp = FakeSpan(name, span_type=span_type)
        m.spans.append(sp)
        yield sp

    def start_span_no_context(name, **kw):
        sp = FakeSpan(name, **kw)
        m.spans.append(sp)
        return sp

    m.start_span, m.start_span_no_context = start_span, start_span_no_context
    m.update_current_trace = lambda tags: m.tags.update(tags)
    monkeypatch.setitem(sys.modules, "mlflow", m)
    return m


def test_disabled_is_a_no_op():
    assert tracing.span_factory(False) is no_span
    assert tracing.setup("") is False
    with tracing.root("chat turn", False, {"turn_id": "x"}) as r:
        r.set_inputs({"q": 1})
        r.set_attributes({"a": [1]})
        r.set_outputs({"a": 2})


def test_root_and_spans_with_json_attributes(fake_mlflow):
    span = tracing.span_factory(True)
    with tracing.root("chat turn", True, {"turn_id": "t1"}):
        with span("tool next_arrivals", "TOOL") as sp:
            sp.set_attributes({"args": {"n": 5}, "row_count": 5, "names": ["a"]})
    assert [s.name for s in fake_mlflow.spans] == ["chat turn", "tool next_arrivals"]
    assert fake_mlflow.spans[0].kw == {"span_type": "AGENT"} and fake_mlflow.tags == {"turn_id": "t1"}
    assert fake_mlflow.spans[1].attrs == {"args": '{"n": 5}', "row_count": 5, "names": '["a"]'}


def test_a_broken_tracer_never_breaks_the_turn(monkeypatch):
    m = types.ModuleType("mlflow")

    def boom(**kw):
        raise RuntimeError("no workspace")
    m.start_span = boom
    monkeypatch.setitem(sys.modules, "mlflow", m)
    with tracing.span_factory(True)("model round 1", "CHAT_MODEL") as sp:
        sp.set_attributes({"x": 1})        # the no-op span


def test_errors_inside_a_span_propagate(fake_mlflow):
    with pytest.raises(ValueError):
        with tracing.span_factory(True)("tool x", "TOOL"):
            raise ValueError("tool failed")


def test_write_plan_parents_and_real_times(fake_mlflow):
    plan = [{"name": "refresh run 7", "parent": None, "kind": "CHAIN", "start_ms": 1000, "end_ms": 9000,
             "attrs": {"run_id": 7}},
            {"name": "task poll", "parent": 0, "kind": "CHAIN", "start_ms": 1000, "end_ms": 5000, "attrs": {}},
            {"name": "setup", "parent": 1, "kind": "UNKNOWN", "start_ms": 1000, "end_ms": 2000, "attrs": {}}]
    assert tracing.write_plan(plan, {"kind": "refresh"}) == "tr-fake"
    root, task, setup = fake_mlflow.spans
    assert root.kw["parent_span"] is None and task.kw["parent_span"] is root and setup.kw["parent_span"] is task
    assert root.kw["tags"] == {"kind": "refresh"} and task.kw["tags"] is None
    assert (setup.kw["start_time_ns"], setup.end_ns, root.end_ns) == (1_000_000_000, 2_000_000_000, 9_000_000_000)
    assert tracing.write_plan([]) is None
