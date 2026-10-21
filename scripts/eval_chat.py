"""Chat with SkyWatch — golden-set evaluation (docs/CHAT_PLAN.md, roadmap CHAT-EVAL-1). Local,
off-platform, same turn loop as the App (lib.chat.run_turn), real tools, real endpoint.

    python scripts/eval_chat.py --warehouse-id <id> [--endpoint ...] [--only live-01,oos-02]

Per item: tool selection (expected tools called, forbidden ones not), grounded answer (the loop's
own check passed — not the fallback), refusal (out-of-scope items), mentions (the answer quotes a
value from the named tool's result), latency and tokens. Writes every turn to a JSONL file.

`explain_eta` needs the in-process model, which lives in the App; here it returns an error, so
those items measure tool selection only. `ask_genie` is real (Genie through managed MCP, ~20 s).

Quota-aware (roadmap OPT-4, src/lib/eval_cache.py): a full run whose inputs (turn loop, grounding,
tool SQL, golden set, this script, endpoint) match the last full run prints that result and stops
— no warehouse or model call; `--force` re-runs. Tool results are cached on disk while the data
version (the live picture's `scored_at`) and the tool SQL don't change, so a re-run on frozen
data pays only for model calls; `--fresh-tools` ignores the cache.
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
from lib import eval_cache  # noqa: E402
from lib.chat import (  # noqa: E402
    APP_TOOLS, SNAPSHOT, ToolCall, compact, genie_ask, run_turn, uc_call_sql, uc_tool_specs,
)

CACHE_DIR = ROOT / ".eval_cache"                    # git-ignored
FINGERPRINT_FILES = [ROOT / "src" / "lib" / "chat.py", ROOT / "src" / "lib" / "grounding.py",
                     ROOT / "src" / "chat_tools.sql", Path(__file__).resolve()]
GENIE_SPACE = "01f1a7adfd07143d8c92fd72fa8fa48f"   # "SkyWatch — KATL Arrival Manager"
REFUSAL = ("can't", "cannot", "can not", "unable", "not able", "only help", "can only", "i can answer",
           "outside", "not something i", "i'm sorry", "i am sorry", "i can help with", "i'm skywatch",
           "don't have the ability", "can't share", "cannot share", "not able to share")


class Tools:
    def __init__(self, ws: dbx.Workspace, wh: str, cache: eval_cache.ToolCache):
        self.ws, self.wh, self.cache = ws, wh, cache
        self.sql_calls = 0
        r = cache.get("_routines", {}, lambda: self._sql(
            "SELECT routine_name, comment FROM skywatch.information_schema.routines WHERE routine_schema = 'chat'"))
        p = cache.get("_parameters", {}, lambda: self._sql(
            "SELECT specific_name, parameter_name, data_type, ordinal_position "
            "FROM skywatch.information_schema.parameters WHERE specific_schema = 'chat'"))
        self.specs = uc_tool_specs(r, p) + APP_TOOLS
        self.by_name = {s.name: s for s in self.specs}

    def _sql(self, statement: str) -> list[dict]:
        self.sql_calls += 1
        return self.ws.sql(self.wh, statement).to_dict("records")

    def execute(self, name: str, args: dict):
        return self.cache.get(name, args, lambda: self._execute(name, args))

    def _execute(self, name: str, args: dict):
        spec = self.by_name[name]
        if name == "explain_eta":
            raise RuntimeError("explain_eta runs only inside the App (the model is loaded there)")
        if name == "ask_genie":
            return genie_ask(lambda path, body: self.ws.post(path, body, timeout=120),
                             GENIE_SPACE, str(args.get("question", "")))
        self.sql_calls += 1
        d = self.ws.post("/api/2.0/sql/statements", {"warehouse_id": self.wh, "statement": uc_call_sql(spec, args),
                                                     "wait_timeout": "50s"})
        if d["status"]["state"] != "SUCCEEDED":
            raise RuntimeError(d["status"].get("error", {}).get("message", d["status"]["state"])[:200])
        cols = [c["name"] for c in d["manifest"]["schema"]["columns"]]
        return compact(cols, (d.get("result") or {}).get("data_array") or [])


def fetch_snapshot(tools: Tools) -> tuple[list[ToolCall], float]:
    """The live snapshot, fetched in parallel like the App does (it caches it for 60 s, so in a
    demo most turns reuse it — the eval fetches once and reuses it for every item)."""
    t0 = time.time()

    def one(name_args):
        name, args = name_args
        c = ToolCall(name, dict(args), prefetched=True)
        try:
            c.result = tools.execute(name, args)
        except Exception as exc:  # noqa: BLE001
            c.error = str(exc)[:200]
        return c
    with ThreadPoolExecutor(len(SNAPSHOT)) as pool:
        calls = list(pool.map(one, SNAPSHOT))
    return calls, time.time() - t0


def data_version(ws: dbx.Workspace, wh: str) -> str:
    """One cheap probe: when Model 1 last scored the live picture. Every refresh moves it."""
    d = ws.post("/api/2.0/sql/statements", {"warehouse_id": wh, "wait_timeout": "50s",
                                            "statement": "SELECT max(scored_at) FROM skywatch.chat.arrival_picture()"})
    if d["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(d["status"].get("error", {}).get("message", d["status"]["state"])[:200])
    return str(((d.get("result") or {}).get("data_array") or [[None]])[0][0])


MODEL_CALLS = [0]


def llm_for(ws: dbx.Workspace, endpoint: str):
    def llm(messages, schemas):
        MODEL_CALLS[0] += 1
        return ws.post(f"/serving-endpoints/{endpoint}/invocations",
                       {"messages": messages, "tools": schemas, "temperature": 0, "max_tokens": 700})
    return llm


def score(item: dict, turns: list, mention_value: str | None) -> dict:
    t = turns[-1]
    called = {c.name for tr in turns for c in tr.calls}
    if "*" in item.get("forbid_tools", []):   # the snapshot is there for every turn; only real calls count
        called = {c.name for tr in turns for c in tr.calls if not c.prefetched}
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


def run_item(item: dict, tools: Tools, llm, snapshot: list[ToolCall]) -> dict:
    t0 = time.time()
    first = run_turn(item["question"], llm=llm, execute=tools.execute, tools=tools.specs, prefetched=snapshot)
    turns = [first]
    if item.get("followup"):
        turns.append(run_turn(item["followup"], llm=llm, execute=tools.execute, tools=tools.specs,
                              history=[first], prefetched=snapshot))
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
    ap.add_argument("--snapshot", action="store_true", help="prefetch the live snapshot (roadmap OPT-1)")
    ap.add_argument("--force", action="store_true", help="run even if nothing changed since the last full run")
    ap.add_argument("--fresh-tools", action="store_true", help="ignore the tool-result cache (still writes it)")
    args = ap.parse_args()

    fp = eval_cache.fingerprint(FINGERPRINT_FILES + [Path(args.golden)],
                                [args.endpoint, f"snapshot={args.snapshot}"])
    stamp_file = CACHE_DIR / "last_run.json"
    if eval_cache.should_skip(eval_cache.load_stamp(stamp_file), fp, force=args.force, partial=bool(args.only)):
        print(f"Nothing changed since the last full run (fingerprint {fp}) — not re-running; --force to "
              f"run anyway. That run:\n{eval_cache.load_stamp(stamp_file)['summary']}")
        return

    ws = dbx.Workspace(args.profile)
    version = data_version(ws, args.warehouse_id)
    cache = eval_cache.ToolCache(CACHE_DIR / "tools.json", version,
                                 eval_cache.fingerprint([ROOT / "src" / "chat_tools.sql"]), reuse=not args.fresh_tools)
    tools = Tools(ws, args.warehouse_id, cache)
    nxt = tools.execute("next_arrivals", {"n": 5})
    cs = [r[nxt["columns"].index("callsign")] for r in nxt["rows"]]
    items = json.load(open(args.golden))["items"]
    if args.only:
        items = [i for i in items if i["id"] in args.only.split(",")]
    for i in items:
        for k in ("question", "followup"):
            if i.get(k):
                i[k] = i[k].replace("{cs0}", cs[0] if cs else "DAL1").replace("{cs1}", cs[1] if len(cs) > 1 else "DAL2")
    snapshot, snap_s = fetch_snapshot(tools) if args.snapshot else ([], 0.0)
    print(f"{len(items)} items · {len(tools.specs)} tools · endpoint {args.endpoint} · callsigns {cs[:2]}"
          + (f" · snapshot {len(snapshot)} calls in {snap_s:.1f} s" if args.snapshot else ""))

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda i: run_item(i, tools, llm_for(ws, args.endpoint), snapshot), items))
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    cache.save()

    def rate(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "—"
    secs = sorted(r["seconds"] for r in results)
    summary = (f"\n| metric | value |\n|---|---|\n| pass (all checks) | {rate('pass')} |\n| tool selection | {rate('tools_ok')} |"
          f"\n| grounded answer (no fallback) | {rate('grounded')} |\n| refusal on out-of-scope | {rate('refusal_ok')} |"
          f"\n| mentions tool value | {rate('mentions_ok')} |\n| nudged / repaired | "
          f"{sum(r['nudged'] for r in results)} / {sum(r['repaired'] for r in results)} |"
          f"\n| latency p50 / p95 s | {statistics.median(secs):.1f} / {secs[min(len(secs) - 1, int(0.95 * len(secs)))]:.1f} |"
          f"\n| tokens per item p50 | {statistics.median(r['tokens'] for r in results):.0f} |"
          f"\n| model rounds per item p50 / mean | {statistics.median(r['rounds'] for r in results):.0f} / "
          f"{statistics.mean(r['rounds'] for r in results):.2f} |"
          f"\n| compute: tool SQL / cache hits / model calls | {tools.sql_calls + 1} / {cache.hits} / {MODEL_CALLS[0]} |")
    print(f"data version {version} · tool cache {'reused' if cache.reused else 'new'}" + summary)
    if not args.only:
        eval_cache.save_stamp(stamp_file, fp, summary)
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
