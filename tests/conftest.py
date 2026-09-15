"""Shared pytest fixtures. Requires `pyspark` + a JDK (see requirements-dev.txt) — CI installs
both; locally, `pip install -r requirements-dev.txt` plus any JDK 11/17 on PATH works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# PySpark spawns workers using PYSPARK_PYTHON (falling back to whatever `python3` resolves to
# on PATH) — in a venv that's not necessarily the same interpreter running pytest, and a
# driver/worker Python minor-version mismatch is a hard error. Pin both to sys.executable.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("skywatch-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
