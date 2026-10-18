# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — LLM ops briefing
# MAGIC
# MAGIC One short plain-English paragraph for the arrival coordinator, built from what the three
# MAGIC models already publish (docs/OPS_BRIEFING_PLAN.md). Runs a few minutes behind the
# MAGIC scoring jobs and appends **one row per run** to `skywatch.stream.ops_briefings` — LLM or
# MAGIC fallback, pass or fail — so every briefing the App ever showed can be audited later.
# MAGIC
# MAGIC 1. latest `predictions` / `demand_forecast` / `irregularity_flags` → `lib.briefing.build_facts`
# MAGIC 2. Foundation Model API chat endpoint (pay-per-token) → `lib.briefing.check_briefing`
# MAGIC 3. endpoint error or check fails → `lib.briefing.template_briefing` (same facts, f-strings)
# MAGIC
# MAGIC Data age is stored as columns and rendered by the App — the model never sees it.

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    # blank = same as stream_schema — see the note in score_eta.py / docs/MLOPS_PLAN.md Track 3.
    dbutils.widgets.text("read_stream_schema", "")
    dbutils.widgets.text("apt_icao", "KATL")
    dbutils.widgets.text("aar_per_hour", "90")
    dbutils.widgets.text("endpoint", "databricks-llama-4-maverick")
    dbutils.widgets.text("temperature", "0.2")
    dbutils.widgets.text("max_tokens", "1024")       # reasoning models spend tokens before the answer
    dbutils.widgets.text("stale_after_min", "30")
    STREAM = dbutils.widgets.get("stream_schema")
    READ_STREAM = dbutils.widgets.get("read_stream_schema").strip() or STREAM
    APT_ICAO = dbutils.widgets.get("apt_icao")
    AAR = float(dbutils.widgets.get("aar_per_hour"))
    ENDPOINT = dbutils.widgets.get("endpoint").strip()
    TEMPERATURE = float(dbutils.widgets.get("temperature"))
    MAX_TOKENS = int(dbutils.widgets.get("max_tokens"))
    STALE_AFTER = int(dbutils.widgets.get("stale_after_min"))
except Exception:
    STREAM, APT_ICAO, AAR = "skywatch.stream", "KATL", 90.0
    READ_STREAM = STREAM
    ENDPOINT, TEMPERATURE, MAX_TOKENS, STALE_AFTER = "databricks-llama-4-maverick", 0.2, 1024, 30

print(f"ops briefing for {APT_ICAO} from {READ_STREAM} | endpoint: {ENDPOINT or '(none — template only)'}")

# COMMAND ----------
# MAGIC %md ## 1. Latest rows from each source → facts

# COMMAND ----------
import datetime as dt
import json

from lib.briefing import build_facts, build_messages, check_briefing, extract_text, template_briefing


def latest(table: str, cols: str, where: str = "1 = 1"):
    """Newest scoring run of `table` as pandas, or None when the table doesn't exist yet
    (the source is then 'unavailable' in the briefing, not 'zero')."""
    name = f"{READ_STREAM}.{table}"
    if not spark.catalog.tableExists(name):
        return None
    return spark.sql(f"""
        SELECT {cols} FROM {name}
        WHERE {where} AND scored_at = (SELECT max(scored_at) FROM {name} WHERE {where})
    """).toPandas()


preds = latest("predictions", "callsign, icao, predicted_eta_min, scored_at")
fc = latest("demand_forecast",
            "bin_start_ts, horizon_step, arrivals_so_far, predicted_mean, predicted_q90, scored_at")
flags = latest("irregularity_flags", "kind, severity, callsign, icao, dist_to_apt_nm, scored_at",
               where=f"apt_icao = '{APT_ICAO}'")

now = dt.datetime.now(dt.timezone.utc)
facts = build_facts(preds, fc, flags, apt_icao=APT_ICAO, aar_per_hour=AAR, now=now,
                    stale_after_min=STALE_AFTER)
print(json.dumps(facts, indent=1))

# COMMAND ----------
# MAGIC %md ## 2. LLM → grounding check → fallback

# COMMAND ----------
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

w = WorkspaceClient()


def ask(endpoint: str, retries: int = 4) -> dict:
    """Raw `llm/v1/chat` response. Pay-per-token endpoints have a per-workspace QPS limit
    (REQUEST_LIMIT_EXCEEDED) — back off and retry that one; anything else goes to the fallback."""
    body = {"messages": build_messages(facts), "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    for attempt in range(retries + 1):
        try:
            return w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)
        except DatabricksError as exc:
            if getattr(exc, "error_code", "") != "REQUEST_LIMIT_EXCEEDED" or attempt == retries:
                raise
            time.sleep(2 ** attempt)


llm_text, reasons, usage, latency_ms = None, [], {}, None
if ENDPOINT:
    t0 = time.time()
    try:
        resp = ask(ENDPOINT)
        latency_ms = int((time.time() - t0) * 1000)
        usage = resp.get("usage") or {}
        llm_text = extract_text(resp)
        check = check_briefing(llm_text, facts)
        reasons = check.reasons
    except Exception as exc:  # noqa: BLE001 — any endpoint failure means "use the template"
        latency_ms = int((time.time() - t0) * 1000)
        reasons = [f"endpoint call failed: {type(exc).__name__}: {str(exc)[:300]}"]
else:
    reasons = ["no endpoint configured"]

if llm_text and not reasons:
    briefing, source = llm_text, "llm"
else:
    briefing, source = template_briefing(facts), "template"
    assert check_briefing(briefing, facts).passed, "template failed its own check"  # tested; belt and braces

print(f"[{source}] {briefing}")
if reasons:
    print("LLM answer rejected:", reasons)

# COMMAND ----------
# MAGIC %md ## 3. Append the audit row

# COMMAND ----------
data_as_of = min((d["as_of"] for d in facts["data"].values() if d), default=None)
row = {
    "generated_at": now,
    "apt_icao": APT_ICAO,
    "briefing": briefing,
    "source": source,
    "endpoint": ENDPOINT or None,
    "llm_text": llm_text,
    "check_passed": source == "llm",
    "check_reasons": " | ".join(reasons) or None,
    "facts_json": json.dumps(facts),
    "data_as_of": (dt.datetime.strptime(data_as_of, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
                   if data_as_of else None),
    "max_age_min": facts["max_age_min"],
    "source_skew_min": facts["source_skew_min"],
    "stale": facts["stale"],
    "latency_ms": latency_ms,
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
}
schema = """generated_at TIMESTAMP, apt_icao STRING, briefing STRING, source STRING, endpoint STRING,
            llm_text STRING, check_passed BOOLEAN, check_reasons STRING, facts_json STRING,
            data_as_of TIMESTAMP, max_age_min INT, source_skew_min INT, stale BOOLEAN,
            latency_ms INT, prompt_tokens INT, completion_tokens INT"""
# plain Python values in schema order — no pandas dtype inference on a one-row frame
(spark.createDataFrame([tuple(row.values())], schema=schema)
      .write.mode("append").option("mergeSchema", "true")
      .saveAsTable(f"{STREAM}.ops_briefings"))
print(f"appended to {STREAM}.ops_briefings")
