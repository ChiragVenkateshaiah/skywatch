"""The App's copies of the shared modules (app/lib/) must match src/lib/ exactly — see
scripts/sync_app_lib.py for why they are copies. Fix a failure with: python scripts/sync_app_lib.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sync_app_lib  # noqa: E402


@pytest.mark.parametrize("name", sync_app_lib.SHARED)
def test_app_copy_matches_src(name):
    copy = ROOT / "app" / "lib" / name
    assert copy.exists(), f"app/lib/{name} missing — run python scripts/sync_app_lib.py"
    assert copy.read_text() == sync_app_lib.expected(name), (
        f"app/lib/{name} has drifted from src/lib/{name} — run python scripts/sync_app_lib.py")
