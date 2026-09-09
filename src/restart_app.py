# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Arrival Manager App keep-alive
# MAGIC
# MAGIC Free Edition auto-stops a Databricks App ~24 h after it starts. This job starts the app
# MAGIC again if it isn't running, so it stays up across a demo window. Scheduled by
# MAGIC `resources/skywatch.appkeepalive.job.yml`, **deployed PAUSED** — unpause it (or just run it
# MAGIC on demand) only while you actually want the app live, since a running app consumes compute.

# COMMAND ----------
try:
    dbutils.widgets.text("app_name", "skywatch-arrival-manager")
    APP_NAME = dbutils.widgets.get("app_name")
except Exception:
    APP_NAME = "skywatch-arrival-manager"

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
app = w.apps.get(name=APP_NAME)
state = getattr(getattr(app, "compute_status", None), "state", None)
print(f"{APP_NAME}: compute state = {state}")

if str(state) not in ("ComputeState.ACTIVE", "ACTIVE"):
    print("starting…")
    w.apps.start(name=APP_NAME).result()
    print("started")
else:
    print("already active — nothing to do")
