"""Great-circle geometry as PySpark Column expressions.

A faithful extraction of the math `pipeline_medallion.py` computes inline (its own copy is
still the one actually running in the deployed DLT pipeline — see the note in that file).
Pulled out here so it has a real pytest suite (`tests/test_geometry.py`) instead of only
ever being exercised by a live pipeline run. `pipeline_medallion.py` and `build_gold.py`
will import from here once Track 3 (catalog-isolated dev target) gives us somewhere safe
to verify the change against real data before it touches the live pipeline.

Every function takes column names (str) or `Column` expressions and returns a `Column` —
call them inside a `.withColumn(...)` / `.select(...)`, same as any `pyspark.sql.functions`
helper. No `spark` / `dbutils` reference anywhere, so this module is plain-importable.
"""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

EARTH_RADIUS_NM = 3440.065


def _cols(*names: str | Column) -> tuple[Column, ...]:
    return tuple(F.col(n) if isinstance(n, str) else n for n in names)


def haversine_nm(
    lat1: str | Column, lon1: str | Column, lat2: str | Column, lon2: str | Column
) -> Column:
    """Great-circle distance between two points, in nautical miles."""
    lat1, lon1, lat2, lon2 = _cols(lat1, lon1, lat2, lon2)
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dphi = F.radians(lat2 - lat1)
    dlam = F.radians(lon2 - lon1)
    a = F.sin(dphi / 2) ** 2 + F.cos(p1) * F.cos(p2) * F.sin(dlam / 2) ** 2
    return F.lit(EARTH_RADIUS_NM) * 2 * F.asin(F.sqrt(a))


def initial_bearing_deg(
    lat1: str | Column, lon1: str | Column, lat2: str | Column, lon2: str | Column
) -> Column:
    """Initial great-circle bearing from point 1 to point 2, degrees 0..360."""
    lat1, lon1, lat2, lon2 = _cols(lat1, lon1, lat2, lon2)
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dlam = F.radians(lon2 - lon1)
    y = F.sin(dlam) * F.cos(p2)
    x = F.cos(p1) * F.sin(p2) - F.sin(p1) * F.cos(p2) * F.cos(dlam)
    return (F.degrees(F.atan2(y, x)) + 360) % 360


def angular_diff_deg(a: str | Column, b: str | Column) -> Column:
    """Smallest absolute difference between two bearings, degrees 0..180."""
    a, b = _cols(a, b)
    d = F.abs((a - b) % 360)
    return F.least(d, 360 - d)
