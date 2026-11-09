import pandas as pd

from temporal_leak import empty_periods, observable


def test_empty_days_inside_the_range_are_found():
    t = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 23:00", "2026-01-04 08:00"])
    assert empty_periods(t) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]


def test_explicit_bounds_extend_the_range():
    t = pd.to_datetime(["2026-01-02"])
    assert empty_periods(t, start="2026-01-01", end="2026-01-03") == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03")]


def test_censored_and_empty_rows_are_marked_not_labelled():
    starts = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"])
    r = observable(starts, "1D", data_end="2026-01-04", empty=[pd.Timestamp("2026-01-03")])
    assert list(r) == ["", "empty", "censored"]
