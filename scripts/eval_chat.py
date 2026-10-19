"""Chat with SkyWatch — golden-set evaluation (docs/CHAT_PLAN.md, roadmap CHAT-EVAL-1). Local,
off-platform, same turn loop as the App (lib.chat.run_turn), real tools, real endpoint.

    python scripts/eval_chat.py --warehouse-id <id> [--endpoint ...] [--only live-01,oos-02]

Per item: tool selection (expected tools called, forbidden ones not), grounded answer (the loop's
own check passed — not the fallback), refusal (out-of-scope items), mentions (the answer quotes a
value from the named tool's result), latency and tokens. Writes every turn to a JSONL file.

`explain_eta` needs the in-process model, which lives in the App; here it returns an error, so
those items measure tool selection only. `ask_genie` is real (Genie through managed MCP, ~20 s).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbx  # noqa: E402
from lib.chat import APP_TOOLS, compact, genie_ask, run_turn, uc_call_sql, uc_tool_specs  # noqa: E402

GENIE_SPACE = "01f1a7adfd07143d8c92fd72fa8fa48f"   # "SkyWatch — KATL Arrival Manager"
REFUSAL = ("can't", "cannot", "can not", "unable", "not able", "only help", "can only", "i can answer",
           "outside", "not something i", "i'm sorry", "i am sorry", "i can help with", "i'm skywatch")


class Tools:
    def __init__(self, ws: dbx.Workspace, wh: str):
        self.ws, self.wh = ws, wh
        r = ws.sql(wh, "SELECT routine_name, comment FROM skywatch.information_schema.routines "
                       "WHERE routine_schema = 'chat'")
        p = ws.sql(wh, "SELECT specific_name, parameter_name, data_type, ordinal_position "
                       "FROM skywatch.information_schema.parameters WHERE specific_schema = 'chat'")
        self.specs = uc_tool_specs(r.to_dict("records"), p.to_dict("records")) + APP_TOOLS
        self.by_name = {s.name: s for s in self.specs}

    def execute(self, name: str, args: dict):
        spec = self.by_name[name]
        if name == "explain_eta":
            raise RuntimeError("explain_eta runs only inside the App (the model is loaded there)")
        if name == "ask_genie":
            return genie_ask(lambda path, body: self.ws.post(path, body, timeout=120),
                             GENIE_SPACE, str(args.get("question", "")))
        d = self.ws.post("/api/2.0/sql/statements", {"warehouse_id": self.wh, "statement": uc_call_sql(spec, args),
                                                     "wait_timeout": "50s"})
        if d["status"]["state"] != "SUCCEEDED":
            raise RuntimeError(d["status"].get("error", {}).get("message", d["status"]["state"])[:200])
        cols = [c["name"] for c in d["manifest"]["schema"]["columns"]]
        return compact(cols, (d.get("result") or {}).get("data_array") or [])


def llm_for(ws: dbx.Workspace, endpoint: str):
    def llm(messages, schemas):
        return ws.post(f"/serving-endpoints/{endpoint}/invocations",
                       {"messages": messages, "tools": schemas, "temperature": 0, "max_tokens": 700})
    return llm


def score(item: dict, turns: list, mention_value: str | None) -> dict:
    t = turns[-1]
    called = {c.name for tr in turns for c in tr.calls}
    tools_ok = True
    for want in item.get("expect_tools", []):
        tools_ok &= any(w in called for w in want.split("|"))
    if "*" in item.get("forbid_tools", []):
        tools_ok &= not called
    else:
        tools_ok &= not (called & set(item.get("forbid_tools", [])))
    out = {"tools_ok": tools_ok, "grounded": None if item.get("app_only") else t.source == "llm"}
    if item.get("refusal"):
        out["refusal_ok"] = any(m in t.answer.lower() for m in REFUSAL)
    if item.get("mentions"):
        out["mentions_ok"] = None if mention_value is None else mention_value.lower() in t.answer.lower()
    if item.get("must_contain_any"):
        out["contain_ok"] = any(m.lower() in t.answer.lower() for m in item["must_contain_any"])
    out["pass"] = all(v for v in out.values() if v is not None)
    return out


def mention_value(item: dict, turns: list) -> str | None:
    m = item.get("mentions")
    if not m:
        return None
    for c in (c for tr in turns for c in tr.calls):
        if c.name == m["tool"] and isinstance(c.result, dict) and c.result.get("rows"):
            cols, rows = c.result["columns"], c.result["rows"]
            if m["column"] in cols and len(rows) > m["row"]:
                return str(rows[m["row"]][cols.index(m["column"])])
    return None


def run_item(item: dict, tools: Tools, llm) -> dict:
    t0 = time.time()
    first = run_turn(item["question"], llm=llm, execute=tools.execute, tools=tools.specs)
    turns = [first]
    if item.get("followup"):
        turns.append(run_turn(item["followup"], llm=llm, execute=tools.execute, tools=tools.specs, history=[first]))
    s = score(item, turns, mention_value(item, turns))
    t = turns[-1]
    return {"id": item["id"], **s, "seconds": round(time.time() - t0, 1), "rounds": sum(x.rounds for x in turns),
            "tokens": sum(x.prompt_tokens + x.completion_tokens for x in turns),
            "question": item["question"] + (f" -> {item['followup']}" if item.get("followup") else ""),
            "tools": [f"{c.name}({json.dumps(c.args)})" + (f" ERR {c.error}" if c.error else "") for x in turns for c in x.calls],
            "answer": t.answer, "check_reasons": t.check.reasons, "nudged": any(x.nudged for x in turns),
            "repaired": any(x.repaired for x in turns),
            "repair_reasons": [r for x in turns for r in x.repair_reasons]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warehouse-id", required=True)
    ap.add_argument("--profile", default="skywatch")
    ap.add_argument("--endpoint", default="databricks-llama-4-maverick")
    ap.add_argument("--golden", default=str(ROOT / "tests" / "chat_golden.json"))
    ap.add_argument("--only", default="", help="comma-separated item ids")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="eval_chat.jsonl")
    args = ap.parse_args()

    ws = dbx.Workspace(args.profile)
    tools = Tools(ws, args.warehouse_id)
    nxt = tools.execute("next_arrivals", {"n": 5})
    cs = [r[nxt["columns"].index("callsign")] for r in nxt["rows"]]
    items = json.load(open(args.golden))["items"]
    if args.only:
        items = [i for i in items if i["id"] in args.only.split(",")]
    for i in items:
        for k in ("question", "followup"):
            if i.get(k):
                i[k] = i[k].replace("{cs0}", cs[0] if cs else "DAL1").replace("{cs1}", cs[1] if len(cs) > 1 else "DAL2")
    print(f"{len(items)} items · {len(tools.specs)} tools · endpoint {args.endpoint} · callsigns {cs[:2]}")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda i: run_item(i, tools, llm_for(ws, args.endpoint)), items))
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def rate(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "—"
    secs = sorted(r["seconds"] for r in results)
    print(f"\n| metric | value |\n|---|---|\n| pass (all checks) | {rate('pass')} |\n| tool selection | {rate('tools_ok')} |"
          f"\n| grounded answer (no fallback) | {rate('grounded')} |\n| refusal on out-of-scope | {rate('refusal_ok')} |"
          f"\n| mentions tool value | {rate('mentions_ok')} |\n| nudged / repaired | "
          f"{sum(r['nudged'] for r in results)} / {sum(r['repaired'] for r in results)} |"
          f"\n| latency p50 / p95 s | {statistics.median(secs):.1f} / {secs[min(len(secs) - 1, int(0.95 * len(secs)))]:.1f} |"
          f"\n| tokens per item p50 | {statistics.median(r['tokens'] for r in results):.0f} |")
    fails = [r for r in results if not r["pass"]]
    print(f"\n{len(fails)} failing items:")
    for r in fails:
        flags = [k for k in ("tools_ok", "grounded", "refusal_ok", "mentions_ok", "contain_ok") if r.get(k) is False]
        print(f"- {r['id']} [{', '.join(flags)}] tools={r['tools']}\n    {r['answer'][:220]}"
              + (f"\n    check: {r['check_reasons'][:3]}" if r["check_reasons"] else ""))
    rr = [x for r in results for x in r["repair_reasons"]]
    if rr:
        from collections import Counter
        print("\nwhy first answers needed a repair:", Counter(x.split(" not in")[0] for x in rr).most_common(8))
    print(f"\nall turns -> {args.out}")


if __name__ == "__main__":
    main()
