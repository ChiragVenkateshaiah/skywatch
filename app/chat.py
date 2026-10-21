"""Chat with SkyWatch — the App's chat page (docs/CHAT_PLAN.md, roadmap CHAT-1, App v2.0).

The turn loop and every guard are lib.chat.run_turn (a copy of src/lib/chat.py — see
scripts/sync_app_lib.py), the same code the golden set evaluates. This page supplies:
  * the LLM call — the chat endpoint, falling back to the second endpoint on an error (OPT-3)
  * the tools — UC functions in skywatch.chat by direct SQL, explain_eta from the in-process
    champion, ask_genie through Databricks-managed MCP, search_docs over the docs index (CHAT-2)
  * the UI — progress while tools run, the data-age label, "How I answered", 👍/👎
  * the audit — one parameterised row per turn in skywatch.chat.chat_audit
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import uuid

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

import data
import model as m1
from concurrent.futures import ThreadPoolExecutor

from lib import docs_index
from lib.chat import (
    APP_TOOLS, PROMPT_VERSION, SNAPSHOT, ToolCall, compact, data_as_of, genie_ask, run_turn, uc_call_sql,
    uc_tool_specs,
)

CHAT_ENDPOINT = os.environ.get("SKYWATCH_CHAT_ENDPOINT", "databricks-llama-4-maverick")
FALLBACK_ENDPOINT = os.environ.get("SKYWATCH_CHAT_FALLBACK_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
GENIE_SPACE = os.environ.get("SKYWATCH_GENIE_SPACE_ID", "")
SELFCHECK_FILE = os.environ.get("SKYWATCH_SELFCHECK_FILE", "/tmp/skywatch_selfcheck.json")
APT_ICAO = os.environ.get("SKYWATCH_APT_ICAO", "KATL")

SUGGESTED = [
    "How busy is KATL right now?",
    "Which aircraft lands next?",
    "Is there any emergency or holding right now?",
    "Is SkyWatch ready, and how fresh is the data?",
    "Will arrivals exceed capacity in the next three hours?",
    "Did the Transformer beat LightGBM?",
    "Which hour of the day is busiest on average?",
]

# Model 1's features in plain words — explain_eta returns these, never the column names
FEATURE_LABELS = {
    "dist_to_apt_nm": "distance to the airport", "heading_err_deg": "heading away from the airport",
    "bearing_sin": "direction it is coming from (east-west)", "bearing_cos": "direction it is coming from (north-south)",
    "alt_ft": "altitude", "gs_kt": "ground speed", "vrate_fpm": "climb or descent rate",
    "closure_kt": "speed towards the airport", "closure_geom_kt": "speed towards the airport (geometric)",
    "turn_rate_dps": "turn rate", "sel_altitude_ft": "selected altitude",
    "airport_inbound_count": "how many other aircraft are inbound", "hour_sin": "time of day",
    "hour_cos": "time of day", "dow_sin": "day of week", "dow_cos": "day of week",
    "is_heavy": "heavy aircraft", "phase": "flight phase",
}

_w = WorkspaceClient()
_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


# --------------------------------------------------------------------------- resources
@st.cache_resource(ttl=3600)
def load_tools():
    rc, rr = data.query_raw("SELECT routine_name, comment FROM skywatch.information_schema.routines "
                            "WHERE routine_schema = 'chat' AND routine_type = 'FUNCTION'")
    pc, pr = data.query_raw("SELECT specific_name, parameter_name, data_type, ordinal_position "
                            "FROM skywatch.information_schema.parameters WHERE specific_schema = 'chat'")
    specs = uc_tool_specs([dict(zip(rc, r)) for r in rr], [dict(zip(pc, r)) for r in pr]) + APP_TOOLS
    if not GENIE_SPACE:
        specs = [s for s in specs if s.name != "ask_genie"]
    return specs


@st.cache_resource(ttl=3600)
def load_docs():
    """The docs index (skywatch.chat.doc_chunks, written by scripts/index_docs.py): ~200 passages,
    searched in memory. Loaded once; search_docs reports it if the table can't be read."""
    try:
        return docs_index.DocIndex.from_rows(data.query_raw(docs_index.LOAD_SQL)[1])
    except Exception as exc:  # noqa: BLE001 — search_docs reports it; the rest of the chat works
        return str(exc)[:200]


def embed(texts: list[str]) -> list[list[float]]:
    return docs_index.embed_response(_w.api_client.do(
        "POST", f"/serving-endpoints/{docs_index.EMBED_ENDPOINT}/invocations", body={"input": texts}))


@st.cache_resource
def load_model():
    try:
        return m1.load_eta_model()
    except Exception as exc:  # noqa: BLE001 — explain_eta reports it; the rest of the chat works
        return (None, str(exc))


@st.cache_data(ttl=60, show_spinner=False)
def live_snapshot() -> list[ToolCall]:
    """The tools that answer most questions, fetched in parallel and shared by every visitor for
    60 s (roadmap OPT-1): a question they cover is answered in one model round instead of two."""
    by_name = {s.name: s for s in load_tools()}

    def one(name_args):
        name, args = name_args
        c = ToolCall(name, dict(args), prefetched=True)
        t0 = time.monotonic()
        try:
            c.result = compact(*data.query_raw(uc_call_sql(by_name[name], args)))
        except Exception as exc:  # noqa: BLE001 — a failed snapshot call is left out, the model can still call it
            c.error = str(exc)[:200]
        c.seconds = time.monotonic() - t0
        return c
    with ThreadPoolExecutor(len(SNAPSHOT)) as pool:
        return list(pool.map(one, [s for s in SNAPSHOT if s[0] in by_name]))


@st.cache_data(ttl=60)
def latest_predictions():
    return data.latest_predictions()


# ------------------------------------------------------------------------------- tools
def explain_eta(callsign: str) -> dict:
    preds = latest_predictions()
    cs = str(callsign or "").strip().upper()
    hit = preds[preds["callsign"].fillna("").str.strip().str.upper() == cs]
    if hit.empty:
        return compact(["note"], [[f"{cs or 'that callsign'} is not in the latest Model 1 run"]])
    mdl, version = load_model()
    if mdl is None:
        raise RuntimeError(f"Model 1 is not loaded in the App: {version}")
    row = hit.iloc[0].to_dict()
    contrib = m1.contributions(mdl, row).head(6)
    rows = [[FEATURE_LABELS.get(f, f.replace("_", " ")), round(float(v), 1)] for f, v in contrib.items()]
    return {"callsign": cs, "predicted_minutes_to_touchdown": round(m1.predict_eta(mdl, row), 1),
            "model_version": str(version), "scored_at": str(row.get("scored_at")),
            **compact(["feature", "minutes_effect (positive = later)"], rows)}


def make_execute(specs, progress):
    by_name = {s.name: s for s in specs}

    def execute(name: str, args: dict):
        progress(f"→ {name}({', '.join(f'{k}={v}' for k, v in args.items())})"
                 + (" — Genie takes about 20 s" if name == "ask_genie" else ""))
        if name == "explain_eta":
            return explain_eta(args.get("callsign", ""))
        if name == "search_docs":
            index = load_docs()
            if isinstance(index, str):
                raise RuntimeError(f"the docs index is not available: {index}")
            return index.search(args.get("query", ""), embed)
        if name == "ask_genie":
            return genie_ask(lambda path, body: _w.api_client.do("POST", path, body=body, headers=_MCP_HEADERS),
                             GENIE_SPACE, str(args.get("question", "")))
        cols, rows = data.query_raw(uc_call_sql(by_name[name], args))
        return compact(cols, rows)
    return execute


def make_llm(used: list):
    """The chat endpoint; on an error, the fallback endpoint for the rest of the turn."""
    def llm(messages, schemas):
        body = {"messages": messages, "tools": schemas, "temperature": 0, "max_tokens": 700}
        for ep in ([used[-1]] if used else [CHAT_ENDPOINT, FALLBACK_ENDPOINT]):
            try:
                r = _w.api_client.do("POST", f"/serving-endpoints/{ep}/invocations", body=body)
                if not used:
                    used.append(ep)
                return r
            except Exception:  # noqa: BLE001
                if ep == FALLBACK_ENDPOINT or used:
                    raise
        raise RuntimeError("no endpoint answered")
    return llm


# ------------------------------------------------------------------------------- audit
def audit(entry: dict) -> None:
    t = entry["turn"]
    try:
        data.execute(
            """INSERT INTO skywatch.chat.chat_audit (turn_id, session_id, asked_at, question, answer, source,
                 tools_json, rounds, nudged, repaired, check_passed, check_reasons, repair_reasons, endpoint,
                 prompt_version, prompt_tokens, completion_tokens, llm_seconds, tool_seconds, total_seconds, data_as_of)
               VALUES (:turn_id, :session_id, current_timestamp(), :question, :answer, :source, :tools_json,
                 CAST(:rounds AS INT), CAST(:nudged AS BOOLEAN), CAST(:repaired AS BOOLEAN), CAST(:passed AS BOOLEAN),
                 :check_reasons, :repair_reasons, :endpoint, :prompt_version, CAST(:pt AS INT), CAST(:ct AS INT),
                 CAST(:llm_s AS DOUBLE), CAST(:tool_s AS DOUBLE), CAST(:total_s AS DOUBLE), :as_of)""",
            {"turn_id": entry["turn_id"], "session_id": st.session_state["session_id"], "question": t.question,
             "answer": t.answer, "source": t.source,
             "tools_json": json.dumps([{"name": c.name, "args": c.args, "error": c.error, "seconds": round(c.seconds, 2),
                                        "rows": (c.result or {}).get("row_count") if isinstance(c.result, dict) else None}
                                       for c in t.calls]),
             "rounds": t.rounds, "nudged": t.nudged, "repaired": t.repaired, "passed": t.check.passed,
             "check_reasons": " | ".join(t.check.reasons) or None,
             "repair_reasons": " | ".join(t.repair_reasons) or None, "endpoint": entry["endpoint"],
             "prompt_version": PROMPT_VERSION, "pt": t.prompt_tokens, "ct": t.completion_tokens,
             "llm_s": round(t.llm_seconds, 2), "tool_s": round(t.tool_seconds, 2),
             "total_s": entry["total_seconds"], "as_of": entry["as_of"]})
    except Exception as exc:  # noqa: BLE001 — never lose the answer because the audit write failed
        entry["audit_error"] = str(exc)[:200]


def save_feedback(turn_id: str, key: str) -> None:
    v = st.session_state.get(key)
    if v is None:
        return
    try:
        data.execute("UPDATE skywatch.chat.chat_audit SET feedback = :fb, feedback_at = current_timestamp() "
                     "WHERE turn_id = :turn_id", {"fb": "up" if v == 1 else "down", "turn_id": turn_id})
    except Exception:  # noqa: BLE001 — feedback is best effort
        pass


# ---------------------------------------------------------------------------- rendering
def _age(iso: str | None) -> str:
    if not iso:
        return ""
    ts = pd.to_datetime(iso, utc=True)
    mins = (dt.datetime.now(dt.timezone.utc) - ts.to_pydatetime()).total_seconds() / 60
    span = f"{mins:.0f} min" if mins < 90 else (f"{mins / 60:.1f} h" if mins < 2880 else f"{mins / 1440:.0f} days")
    return f"data as of {ts:%H:%M} UTC ({span} old)"


def render(entry: dict) -> None:
    t = entry["turn"]
    with st.chat_message("user"):
        st.markdown(t.question)
    with st.chat_message("assistant", avatar="✈️"):
        st.markdown(t.answer)
        badge = ("✅ checked against the tool results" if t.source == "llm"
                 else "⚠️ no checked answer — showing the tool results instead")
        # an answer with no number, time or callsign states no data facts (a refusal, "what does
        # Model 2 do?") — the live snapshot's age would only confuse it
        states_data = any(ch.isdigit() for ch in t.answer)
        parts = [badge, _age(entry["as_of"]) if states_data else "", f"{entry['total_seconds']:.1f} s",
                 entry["endpoint"] if entry["endpoint"] != CHAT_ENDPOINT else ""]
        st.caption(" · ".join(p for p in parts if p))
        with st.expander("How I answered"):
            if not t.calls:
                st.write("No tool was needed.")
            for c in t.calls:
                st.markdown(f"**{c.name}**({', '.join(f'{k}={v!r}' for k, v in c.args.items())}) · "
                            + ("live snapshot, fetched before the question" if c.prefetched else f"{c.seconds:.1f} s"))
                if c.error:
                    st.error(c.error)
                elif isinstance(c.result, dict) and c.result.get("columns"):
                    st.dataframe(pd.DataFrame(c.result["rows"], columns=c.result["columns"]),
                                 hide_index=True, use_container_width=True)
                    if c.result.get("sql"):
                        st.code(c.result["sql"], language="sql")
                    if c.result.get("answer"):
                        st.caption(f"Genie: {c.result['answer']}")
                else:
                    st.json(c.result)
            notes = [f"{t.rounds} model round(s)", f"{t.prompt_tokens + t.completion_tokens} tokens",
                     f"model {t.llm_seconds:.1f} s · tools {t.tool_seconds:.1f} s"]
            if t.nudged:
                notes.append("the model wrote a tool call as text and was asked to use the tool interface")
            if t.repaired:
                notes.append("first answer failed the check (" + "; ".join(t.repair_reasons[:3]) + ") and was repaired")
            if t.source != "llm":
                notes.append("check: " + "; ".join(t.check.reasons[:3]))
            st.caption(" · ".join(notes))
            if entry.get("audit_error"):
                st.caption(f"(audit row not written: {entry['audit_error']})")
        key = f"fb_{entry['turn_id']}"
        st.feedback("thumbs", key=key, on_change=save_feedback, args=(entry["turn_id"], key))


# --------------------------------------------------------------------------------- page
st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
st.session_state.setdefault("chat", [])

with st.sidebar:
    st.subheader("Chat with SkyWatch")
    st.caption(f"Model: `{CHAT_ENDPOINT}` (fallback `{FALLBACK_ENDPOINT}`) · prompt {PROMPT_VERSION}")
    if st.button("New conversation", use_container_width=True):
        st.session_state["chat"] = []
        st.rerun()
    with st.expander("Tool status"):
        checks = None
        try:
            with open(SELFCHECK_FILE) as f:
                checks = json.load(f)
        except (OSError, ValueError):
            pass
        if st.button("Re-check now"):
            import selfcheck
            checks = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      "results": selfcheck.run_checks(genie_query=False)}
        if checks:
            st.caption(f"as the App's service principal, {checks['checked_at']}")
            for r in checks["results"]:
                st.write(f"{'✅' if r['ok'] else '❌'} {r['kind']} · {r['name']}"
                         + ("" if r["ok"] else f" — {r['detail']}"))
        else:
            st.caption("No self-check result yet.")

    # roadmap DATA-2b: the refresh is a human action with a confirmation — it clears the live
    # tables and runs ~10 min. The chat model only gets the read-only refresh_status tool.
    with st.expander("Get SkyWatch ready", expanded=bool(st.session_state.get("refresh_run"))):
        try:
            last = data.query("SELECT * FROM skywatch.chat.refresh_status()")
        except Exception as exc:  # noqa: BLE001
            last = pd.DataFrame()
            st.caption(f"refresh status unavailable: {str(exc)[:120]}")
        if not last.empty:
            r = last.iloc[0]
            icon = {"READY": "🟢", "READY_SLOW": "🟡"}.get(r["status"], "🔴")
            st.write(f"{icon} **{r['status']}** · {r['elapsed_min']:.1f} min · checked {r['checked_at']:%Y-%m-%d %H:%M} UTC")
            st.caption(f"data as of {r['data_as_of']:%H:%M} UTC · {r['inbound']} inbound · {r['flagged']} flagged")
        job_id = os.environ.get("SKYWATCH_REFRESH_JOB_ID", "")
        run_id = st.session_state.get("refresh_run")
        if run_id:
            try:
                run = _w.jobs.get_run(run_id)
                life = run.state.life_cycle_state.value if run.state.life_cycle_state else "?"
                result = run.state.result_state.value if run.state.result_state else ""
                st.write(f"Refresh run {run_id}: **{life}** {result}")
                st.caption(f"[open the run]({run.run_page_url})")
                if life in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
                    st.session_state.pop("refresh_run")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"run status unavailable: {str(exc)[:120]}")
            st.button("Update status", use_container_width=True)
        elif job_id:
            if st.session_state.get("confirm_refresh"):
                st.warning("This clears today's live picture (predictions, demand forecast, flags) and rebuilds "
                           "it from a fresh data burst. It takes about 10 minutes. History is kept.")
                c1, c2 = st.columns(2)
                if c1.button("Yes, refresh", type="primary", use_container_width=True):
                    st.session_state["refresh_run"] = _w.jobs.run_now(int(job_id)).run_id
                    st.session_state.pop("confirm_refresh")
                    st.rerun()
                if c2.button("Cancel", use_container_width=True):
                    st.session_state.pop("confirm_refresh")
                    st.rerun()
            elif st.button("Run refresh…", use_container_width=True):
                st.session_state["confirm_refresh"] = True
                st.rerun()

st.title("💬 Chat with SkyWatch")
st.caption(f"Ask about {APT_ICAO} arrivals and SkyWatch's models. Every number in an answer is checked "
           "against the data the tools returned; times are UTC.")

for e in st.session_state["chat"]:
    render(e)

question = st.chat_input("Ask SkyWatch…") or st.session_state.pop("pending_question", None)  # from the dashboard
if not st.session_state["chat"]:
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 3].button(q, use_container_width=True, key=f"suggest_{i}"):
            question = q

if question:
    try:
        specs = load_tools()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the chat tools from skywatch.chat: {exc}")
        st.stop()
    with st.chat_message("user"):
        st.markdown(question)
    used: list = []
    t0 = time.monotonic()
    with st.status("Looking it up…", expanded=True) as status:
        try:
            turn = run_turn(question, llm=make_llm(used), execute=make_execute(specs, status.write),
                            tools=specs, history=[e["turn"] for e in st.session_state["chat"]],
                            prefetched=live_snapshot())
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001 — endpoint down / warehouse error: say so, keep the page
            status.update(label="Something went wrong", state="error")
            st.error(f"SkyWatch could not answer right now: {type(exc).__name__}: {str(exc)[:200]}")
            st.stop()
    entry = {"turn": turn, "turn_id": uuid.uuid4().hex, "as_of": data_as_of(turn.calls),
             "endpoint": used[-1] if used else CHAT_ENDPOINT, "total_seconds": round(time.monotonic() - t0, 1)}
    audit(entry)
    st.session_state["chat"].append(entry)
    st.rerun()
