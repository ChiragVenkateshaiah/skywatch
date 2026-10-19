-- SkyWatch chat tools (docs/CHAT_PLAN.md §3) — written once as UC functions in skywatch.chat.
-- The App's chat calls them with direct SQL; Databricks-managed MCP
-- (/api/2.0/mcp/functions/skywatch/chat) serves the same functions to any other agent.
-- Each COMMENT is the tool description an MCP client sees, so it is written for the model.
-- Applied by the skywatch_chat_setup job (resources/skywatch.chatsetup.job.yml). Re-runnable.
-- Catalog is literal `skywatch`: this file is a sql_task, and only the dev target uses the chat.

CREATE SCHEMA IF NOT EXISTS skywatch.chat
  COMMENT 'SkyWatch chat tools (UC functions served via Databricks-managed MCP)';

-- ---------------------------------------------------------------------------------- live picture
CREATE OR REPLACE FUNCTION skywatch.chat.arrival_picture()
RETURNS TABLE (inbound INT, within_10_min INT, within_30_min INT, model_version BIGINT, scored_at TIMESTAMP)
COMMENT 'KATL arrival picture from the latest Model 1 run: how many aircraft are inbound, how many are predicted to land within 10 and 30 minutes, and when the run was scored (UTC). No rows means Model 1 has not run.'
RETURN
  SELECT CAST(count(*) AS INT),
         CAST(count_if(predicted_eta_min <= 10) AS INT),
         CAST(count_if(predicted_eta_min <= 30) AS INT),
         max(model_version), max(scored_at)
  FROM skywatch.stream.predictions
  WHERE scored_at = (SELECT max(scored_at) FROM skywatch.stream.predictions)
  HAVING count(*) > 0;

CREATE OR REPLACE FUNCTION skywatch.chat.next_arrivals(n INT)
RETURNS TABLE (rank INT, callsign STRING, ac_type STRING, eta_min DOUBLE, touchdown_utc STRING,
               dist_nm DOUBLE, scored_at TIMESTAMP)
COMMENT 'The next n aircraft predicted to land at KATL, soonest first: callsign, aircraft type, predicted minutes to touchdown, predicted touchdown time (HH:MM UTC), distance in nautical miles. Use n between 1 and 20.'
RETURN
  SELECT * FROM (
    SELECT CAST(row_number() OVER (ORDER BY predicted_touchdown_ts) AS INT) AS rank,
           coalesce(nullif(trim(callsign), ''), icao), ac_type,
           round(predicted_eta_min, 1), date_format(predicted_touchdown_ts, 'HH:mm'),
           round(dist_to_apt_nm, 0), scored_at
    FROM skywatch.stream.predictions
    WHERE scored_at = (SELECT max(scored_at) FROM skywatch.stream.predictions)
  ) WHERE rank <= least(greatest(n, 1), 20);

CREATE OR REPLACE FUNCTION skywatch.chat.aircraft(cs STRING)
RETURNS TABLE (callsign STRING, ac_type STRING, eta_min DOUBLE, touchdown_utc STRING, dist_nm DOUBLE,
               alt_ft DOUBLE, gs_kt DOUBLE, vrate_fpm DOUBLE, phase STRING, scored_at TIMESTAMP)
COMMENT 'One inbound aircraft by callsign (case-insensitive, e.g. DAL1497): predicted minutes to touchdown, touchdown time (HH:MM UTC), distance (nm), altitude (ft), ground speed (kt), vertical rate (ft/min), flight phase. No rows means it is not in the latest Model 1 run.'
RETURN
  SELECT coalesce(nullif(trim(callsign), ''), icao), ac_type, round(predicted_eta_min, 1),
         date_format(predicted_touchdown_ts, 'HH:mm'), round(dist_to_apt_nm, 0), round(alt_ft, 0),
         round(gs_kt, 0), round(vrate_fpm, 0), phase, scored_at
  FROM skywatch.stream.predictions
  WHERE scored_at = (SELECT max(scored_at) FROM skywatch.stream.predictions)
    AND (upper(trim(callsign)) = upper(trim(cs)) OR lower(icao) = lower(trim(cs)));

CREATE OR REPLACE FUNCTION skywatch.chat.demand_outlook(aar_per_hour DOUBLE)
RETURNS TABLE (bin_start_utc STRING, predicted DOUBLE, upper DOUBLE, aar_per_15min DOUBLE,
               over_aar BOOLEAN, upper_over_aar BOOLEAN, next_hour_total DOUBLE, next_3h_total DOUBLE,
               arrivals_seen_by_skywatch_today INT, scored_at TIMESTAMP)
COMMENT 'Model 2 demand forecast for the next 3 hours at KATL, one row per 15-minute bin: predicted arrivals, the upper end of the forecast range, the airport acceptance rate (AAR, capacity) per 15 minutes, and whether the bin is predicted over capacity. next_hour_total / next_3h_total are the predicted arrivals summed over the first 4 / all 12 bins (use them instead of adding bins). arrivals_seen_by_skywatch_today counts only landings SkyWatch observed while collecting data - not the airport total. Pass aar_per_hour = 90 for KATL.'
RETURN
  SELECT date_format(bin_start_ts, 'HH:mm'), round(predicted_mean, 1), round(predicted_q90, 1),
         round(aar_per_hour / 4, 1), predicted_mean > aar_per_hour / 4, predicted_q90 > aar_per_hour / 4,
         round(sum(CASE WHEN horizon_step < 4 THEN predicted_mean END) OVER (), 0),
         round(sum(predicted_mean) OVER (), 0),
         CAST(arrivals_so_far AS INT), scored_at
  FROM skywatch.stream.demand_forecast
  WHERE scored_at = (SELECT max(scored_at) FROM skywatch.stream.demand_forecast)
  ORDER BY horizon_step;

CREATE OR REPLACE FUNCTION skywatch.chat.irregularities()
RETURNS TABLE (kind STRING, kind_count INT, severity INT, callsign STRING, ac_type STRING, dist_nm DOUBLE,
               alt_ft DOUBLE, detail STRING, scored_at TIMESTAMP)
COMMENT 'Aircraft currently flagged by Model 3 rules at KATL: emergency squawk, go-around, or holding pattern, most severe first; kind_count is how many aircraft have that kind of flag. No rows means nothing is flagged right now. Flags from an older run than the current Model 1 picture are not returned.'
RETURN
  SELECT kind, CAST(count(*) OVER (PARTITION BY kind) AS INT), CAST(severity AS INT),
         coalesce(nullif(trim(callsign), ''), icao), ac_type,
         round(dist_to_apt_nm, 0), round(alt_ft, 0), detail, scored_at
  FROM skywatch.stream.irregularity_flags
  WHERE apt_icao = 'KATL'
    AND scored_at = (SELECT max(scored_at) FROM skywatch.stream.irregularity_flags WHERE apt_icao = 'KATL')
    -- score_irregularities writes nothing on a quiet run: same rule as lib/briefing.build_facts
    AND scored_at >= (SELECT max(scored_at) FROM skywatch.stream.predictions) - INTERVAL 15 MINUTES
  ORDER BY severity DESC, dist_to_apt_nm;

CREATE OR REPLACE FUNCTION skywatch.chat.latest_briefing()
RETURNS TABLE (briefing STRING, source STRING, stale BOOLEAN, data_as_of TIMESTAMP, generated_at TIMESTAMP)
COMMENT 'The newest LLM ops briefing for KATL (a short paragraph built from Models 1-3), whether an LLM or the template wrote it, and whether its data was stale.'
RETURN
  SELECT briefing, source, stale, data_as_of, generated_at
  FROM skywatch.stream.ops_briefings
  WHERE apt_icao = 'KATL'
  ORDER BY generated_at DESC
  LIMIT 1;

-- ----------------------------------------------------------------------------------- models
CREATE OR REPLACE FUNCTION skywatch.chat.model_status(model STRING)
RETURNS TABLE (model_name STRING, gate_verdict STRING, challenger_version BIGINT,
               champion_version_before BIGINT, applied BOOLEAN, reasons STRING, audited_at TIMESTAMP)
COMMENT 'Promotion-gate history for one SkyWatch model, newest first (last 5): eta_touchdown (Model 1, time to touchdown; the champion is LightGBM, and Transformer challengers were tested against it) or demand_forecast (Model 2, arrival demand). Each row: PASS or HOLD, which challenger version was tested against which champion version, whether it was applied, and why (overall MAE in minutes vs the champion). Use it for: which model is in production, did a challenger such as the Transformer beat LightGBM, how accurate a model is. No rows means no gate run has been recorded for that model.'
RETURN
  SELECT model_name, gate_verdict, challenger_version, champion_version_before, applied, reasons, audited_at
  FROM skywatch.ml.promotion_audit
  WHERE model_name LIKE concat('%', trim(model))
  ORDER BY audited_at DESC
  LIMIT 5;

CREATE OR REPLACE FUNCTION skywatch.chat.model_health()
RETURNS TABLE (check_type STRING, metric_name STRING, value DOUBLE, severity STRING, detail STRING,
               checked_at TIMESTAMP)
COMMENT 'Latest Model 1 monitoring checks (feature drift PSI per feature, rolling accuracy vs the champion test MAE), most severe first.'
RETURN
  SELECT check_type, metric_name, round(value, 3), severity, detail, checked_at
  FROM skywatch.ml.model_health
  WHERE checked_at = (SELECT max(checked_at) FROM skywatch.ml.model_health)
  ORDER BY CASE severity WHEN 'significant' THEN 0 WHEN 'moderate' THEN 1 ELSE 2 END, metric_name;

-- ------------------------------------------------------------------------------- audit
-- One row per chat turn (roadmap CHAT-EVAL-1). The App appends it after the answer is shown and
-- updates `feedback` when the viewer clicks 👍/👎. App SP: SELECT + MODIFY (skywatch.app.yml).
CREATE TABLE IF NOT EXISTS skywatch.chat.chat_audit (
  turn_id STRING, session_id STRING, asked_at TIMESTAMP,
  question STRING, answer STRING, source STRING,
  tools_json STRING, rounds INT, nudged BOOLEAN, repaired BOOLEAN,
  check_passed BOOLEAN, check_reasons STRING, repair_reasons STRING,
  endpoint STRING, prompt_version STRING,
  prompt_tokens INT, completion_tokens INT,
  llm_seconds DOUBLE, tool_seconds DOUBLE, total_seconds DOUBLE,
  data_as_of STRING, feedback STRING, feedback_at TIMESTAMP
) COMMENT 'Chat with SkyWatch: one row per turn — question, tools, answer, grounding result, latency, tokens, viewer feedback.';
