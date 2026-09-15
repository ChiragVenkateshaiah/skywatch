"""Known-reference-point tests for src/lib/geometry.py.

1 degree of latitude is ~60 nm by the definition of the nautical mile, and due-north /
due-east bearings are easy to get exactly right or exactly wrong — good properties for
catching a sign error or a swapped lat/lon without needing real ADS-B fixtures.
"""

from __future__ import annotations

import pytest

from lib.geometry import angular_diff_deg, haversine_nm, initial_bearing_deg


def _row(spark, **kwargs):
    return spark.createDataFrame([kwargs]).first()


class TestHaversineNm:
    def test_one_degree_latitude_is_about_60nm(self, spark):
        r = _row(spark, lat1=0.0, lon1=0.0, lat2=1.0, lon2=0.0)
        d = spark.createDataFrame([r]).select(
            haversine_nm("lat1", "lon1", "lat2", "lon2").alias("d")
        ).first()["d"]
        assert d == pytest.approx(60.04, abs=0.1)

    def test_same_point_is_zero(self, spark):
        df = spark.createDataFrame([{"lat1": 33.64, "lon1": -84.43, "lat2": 33.64, "lon2": -84.43}])
        d = df.select(haversine_nm("lat1", "lon1", "lat2", "lon2").alias("d")).first()["d"]
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_symmetric(self, spark):
        df = spark.createDataFrame([{"a": 33.64, "b": -84.43, "c": 40.64, "d": -73.78}])
        fwd = df.select(haversine_nm("a", "b", "c", "d").alias("x")).first()["x"]
        rev = df.select(haversine_nm("c", "d", "a", "b").alias("x")).first()["x"]
        assert fwd == pytest.approx(rev, abs=1e-6)


class TestInitialBearingDeg:
    def test_due_north(self, spark):
        df = spark.createDataFrame([{"lat1": 0.0, "lon1": 0.0, "lat2": 1.0, "lon2": 0.0}])
        b = df.select(initial_bearing_deg("lat1", "lon1", "lat2", "lon2").alias("b")).first()["b"]
        assert b == pytest.approx(0.0, abs=1e-6)

    def test_due_east(self, spark):
        df = spark.createDataFrame([{"lat1": 0.0, "lon1": 0.0, "lat2": 0.0, "lon2": 1.0}])
        b = df.select(initial_bearing_deg("lat1", "lon1", "lat2", "lon2").alias("b")).first()["b"]
        assert b == pytest.approx(90.0, abs=1e-6)

    def test_due_south(self, spark):
        df = spark.createDataFrame([{"lat1": 1.0, "lon1": 0.0, "lat2": 0.0, "lon2": 0.0}])
        b = df.select(initial_bearing_deg("lat1", "lon1", "lat2", "lon2").alias("b")).first()["b"]
        assert b == pytest.approx(180.0, abs=1e-6)

    def test_always_in_0_360(self, spark):
        df = spark.createDataFrame([{"lat1": 5.0, "lon1": 5.0, "lat2": 1.0, "lon2": -3.0}])
        b = df.select(initial_bearing_deg("lat1", "lon1", "lat2", "lon2").alias("b")).first()["b"]
        assert 0.0 <= b < 360.0


class TestAngularDiffDeg:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (10.0, 10.0, 0.0),
            (350.0, 10.0, 20.0),   # wraps through 0
            (10.0, 350.0, 20.0),   # symmetric
            (0.0, 180.0, 180.0),   # antipodal bearings, max diff
            (270.0, 0.0, 90.0),
        ],
    )
    def test_known_pairs(self, spark, a, b, expected):
        df = spark.createDataFrame([{"a": a, "b": b}])
        d = df.select(angular_diff_deg("a", "b").alias("d")).first()["d"]
        assert d == pytest.approx(expected, abs=1e-6)
