"""Chat with SkyWatch — permission and tool self-check, run as the App's service principal
(docs/CHAT_PLAN.md §4). Started in the background by app.yaml at every App start, so each
deploy logs, *as the identity that will really use them*, whether every chat dependency works:

    SELFCHECK ok    tool arrival_picture      1 rows   1.9 s
    SELFCHECK FAIL  tool model_status         PERMISSION_DENIED: ...

Read with `databricks apps logs skywatch-arrival-manager | grep SELFCHECK`. `run_checks()` is
also what the chat page's tool-status panel shows.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from databricks.sdk import WorkspaceClient

import data

CHAT_SCHEMA = "skywatch.chat"
# a cheap, valid call for every UC tool (arguments by name)
TOOL_ARGS = {"next_arrivals": "(1)", "aircraft": "('X')", "demand_outlook": "(90.0)", "model_status": "('eta_touchdown')"}

_w = WorkspaceClient()


def _timed(fn):
    t0 = time.monotonic()
    try:
        detail = fn()
        return True, detail, time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001 — a self-check reports, it never raises
        return False, f"{type(exc).__name__}: {str(exc)[:160]}", time.monotonic() - t0


def run_checks(genie_query: bool = True) -> list[dict]:
    out = []

    def add(kind, name, fn):
        ok, detail, secs = _timed(fn)
        out.append({"kind": kind, "name": name, "ok": ok, "detail": detail, "seconds": round(secs, 1)})

    tools = data.query("SELECT routine_name FROM skywatch.information_schema.routines "
                       "WHERE routine_schema = 'chat' AND routine_type = 'FUNCTION'")["routine_name"].tolist()
    for t in sorted(tools):
        add("tool", t, lambda t=t: f"{len(data.query(f'SELECT * FROM {CHAT_SCHEMA}.{t}{TOOL_ARGS.get(t, chr(40) + chr(41))}'))} rows")

    def endpoint(env: str) -> str:
        ep = os.environ.get(env, "")
        if not ep:
            raise RuntimeError(f"{env} not set")
        r = _w.api_client.do("POST", f"/serving-endpoints/{ep}/invocations",
                             body={"messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 5})
        return str(r["choices"][0]["message"]["content"])[:20]

    for env in ("SKYWATCH_CHAT_ENDPOINT", "SKYWATCH_CHAT_FALLBACK_ENDPOINT"):
        add("endpoint", os.environ.get(env, env), lambda env=env: endpoint(env))

    space = os.environ.get("SKYWATCH_GENIE_SPACE_ID", "")
    hdr = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

    def genie():
        if not space:
            raise RuntimeError("SKYWATCH_GENIE_SPACE_ID not set")
        r = _w.api_client.do("POST", f"/api/2.0/mcp/genie/{space}", headers=hdr,
                             body={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = [t["name"] for t in r["result"]["tools"]]
        if not genie_query:
            return f"{len(names)} MCP tools"
        r = _w.api_client.do("POST", f"/api/2.0/mcp/genie/{space}", headers=hdr, body={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": f"query_space_{space}", "arguments": {"query": "How many tables do you have?"}}})
        status = json.loads(r["result"]["content"][0]["text"]).get("status")
        return f"{len(names)} MCP tools, query accepted ({status})"
    add("genie", space or "genie", genie)

    def audit():
        tid = f"selfcheck-{uuid.uuid4().hex[:8]}"
        data.query(f"INSERT INTO {CHAT_SCHEMA}.chat_audit (turn_id, asked_at, source) "
                   f"VALUES ('{tid}', current_timestamp(), 'selfcheck')")
        data.query(f"DELETE FROM {CHAT_SCHEMA}.chat_audit WHERE turn_id = '{tid}'")
        return "insert + delete ok"
    add("audit", "chat_audit", audit)

    return out


if __name__ == "__main__":
    for r in run_checks():
        print(f"SELFCHECK {'ok  ' if r['ok'] else 'FAIL'}  {r['kind']:<8} {r['name']:<38} {r['detail']}  "
              f"{r['seconds']} s", flush=True)
    print("SELFCHECK done", flush=True)
