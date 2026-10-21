"""Index SkyWatch's docs for the chat's search_docs tool (roadmap CHAT-2, docs/CHAT_KNOWLEDGE_PLAN.md).

    python scripts/index_docs.py --warehouse-id <id> [--force] [--dry-run]

Chunks the corpus (lib.docs_index.CORPUS) at its headings, embeds the passages with
databricks-gte-large-en in batches, and replaces skywatch.chat.doc_chunks. Local, not a job: the
bundle doesn't sync *.md, and a serverless job start waited 645-705 s on 2026-10-19 (decided
2026-10-21). Quota-aware like eval_chat.py (OPT-4): if the docs hash matches the table's, it stops
after one query. Run it after a doc change, then restart the App (it loads the index once).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbx  # noqa: E402
from lib import docs_index  # noqa: E402

BATCH_EMBED = 4          # a batch of 16 -> REQUEST_LIMIT_EXCEEDED (workspace QPS limit, 2026-10-21); 4 passed
EMBED_PAUSE_S = 1.0
BATCH_INSERT = 20        # ~20 x 1024 floats per INSERT keeps each statement well under 1 MB


def sql(ws: dbx.Workspace, wh: str, statement: str) -> list[list]:
    d = ws.post("/api/2.0/sql/statements", {"warehouse_id": wh, "statement": statement, "wait_timeout": "50s"})
    if d["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(d["status"].get("error", {}).get("message", d["status"]["state"])[:300])
    return (d.get("result") or {}).get("data_array") or []


def lit(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warehouse-id", required=True)
    ap.add_argument("--profile", default="skywatch")
    ap.add_argument("--force", action="store_true", help="re-index even if the docs are unchanged")
    ap.add_argument("--dry-run", action="store_true", help="chunk and report only; no Databricks call")
    args = ap.parse_args()

    files = {f: (ROOT / f).read_text() for f in docs_index.CORPUS}
    chunks = [c for f, text in files.items() for c in docs_index.chunk_markdown(Path(f).name, text)]
    # hash the passages, not the files: a chunking change must re-index too
    h = docs_index.docs_hash({f"{i:04d} {c['source']}": c["text"] for i, c in enumerate(chunks)})
    print(f"{len(files)} files · {len(chunks)} passages · docs hash {h}")
    if args.dry_run:
        return

    ws = dbx.Workspace(args.profile)
    sql(ws, args.warehouse_id, f"CREATE TABLE IF NOT EXISTS {docs_index.TABLE} (chunk_id INT, source STRING, "
                               "text STRING, embedding STRING, docs_hash STRING, indexed_at TIMESTAMP) "
                               "COMMENT 'Passages of the SkyWatch docs + gte-large-en embeddings (JSON) for the "
                               "chat search_docs tool. Written by scripts/index_docs.py.'")
    current = sql(ws, args.warehouse_id, f"SELECT max(docs_hash), count(*) FROM {docs_index.TABLE}")[0]
    if current[0] == h and not args.force:
        print(f"Index is current ({current[1]} passages, hash {h}) — nothing to do; --force to re-index.")
        return

    t0, vectors = time.time(), []
    for i in range(0, len(chunks), BATCH_EMBED):
        batch = [docs_index.embed_input(c) for c in chunks[i:i + BATCH_EMBED]]
        resp = ws.post(f"/serving-endpoints/{docs_index.EMBED_ENDPOINT}/invocations", {"input": batch})
        vectors += docs_index.embed_response(resp)
        time.sleep(EMBED_PAUSE_S)
    print(f"embedded {len(vectors)} passages in {time.time() - t0:.1f} s "
          f"({-(-len(chunks) // BATCH_EMBED)} endpoint calls)")

    t0 = time.time()
    sql(ws, args.warehouse_id, f"DELETE FROM {docs_index.TABLE}")
    for i in range(0, len(chunks), BATCH_INSERT):
        values = ",\n".join(
            f"({i + j}, {lit(c['source'])}, {lit(c['text'])}, "
            f"{lit(json.dumps([round(x, 6) for x in v], separators=(',', ':')))}, '{h}', current_timestamp())"
            for j, (c, v) in enumerate(zip(chunks[i:i + BATCH_INSERT], vectors[i:i + BATCH_INSERT])))
        sql(ws, args.warehouse_id, f"INSERT INTO {docs_index.TABLE} VALUES\n{values}")
    n = sql(ws, args.warehouse_id, f"SELECT count(*) FROM {docs_index.TABLE} WHERE docs_hash = '{h}'")[0][0]
    print(f"wrote {n} passages to {docs_index.TABLE} in {time.time() - t0:.1f} s — restart the App to load them")


if __name__ == "__main__":
    main()
