import numpy as np
import pandas as pd
import pytest

from temporal_leak import LeakageError, check_before_cutoff, check_fit_rows, check_temporal_split


def dates(*ds):
    return pd.to_datetime(list(ds))


def test_clean_split_passes_and_reports_ranges():
    t = dates("2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03")
    p = ["train", "train", "validation", "test"]
    r = check_temporal_split(t, p)
    assert r["train"]["rows"] == 2 and r["test"]["first"] == pd.Timestamp("2026-01-03")


def test_a_date_in_two_parts_is_a_leak():
    t = dates("2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03")
    with pytest.raises(LeakageError, match="more than one part"):
        check_temporal_split(t, ["train", "train", "test", "test"])


def test_parts_out_of_time_order_are_a_leak():
    t = dates("2026-01-03", "2026-01-01", "2026-01-02")
    with pytest.raises(LeakageError, match="ends at"):
        check_temporal_split(t, ["train", "validation", "test"])


def test_unknown_part_name_is_rejected():
    with pytest.raises(LeakageError, match="unknown"):
        check_temporal_split(dates("2026-01-01"), ["holdout"])


def test_event_after_cutoff_is_a_leak_and_missing_events_are_ignored():
    cut = dates("2026-01-02", "2026-01-02", "2026-01-02")
    assert check_before_cutoff(dates("2026-01-01 23:00", "2026-01-01 01:00", None), cut) == 2
    with pytest.raises(LeakageError, match="at or after"):
        check_before_cutoff(dates("2026-01-01 23:00", "2026-01-02 00:10", None), cut)


def test_event_exactly_at_cutoff_depends_on_strict():
    cut, ev = dates("2026-01-02"), dates("2026-01-02")
    assert check_before_cutoff(ev, cut, strict=False) == 1
    with pytest.raises(LeakageError):
        check_before_cutoff(ev, cut)


def test_fit_rows_outside_the_allowed_rows_are_a_leak():
    train = np.array([True, True, False, False])
    assert check_fit_rows(train, train | np.array([False, False, True, False])) == 2
    with pytest.raises(LeakageError, match="not allowed"):
        check_fit_rows(np.array([True, True, False, True]), train)
