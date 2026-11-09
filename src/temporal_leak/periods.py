"""Periods with no data, and rows whose outcome window cannot be observed."""

from __future__ import annotations

import numpy as np
import pandas as pd


def empty_periods(times, freq: str = "D", start=None, end=None) -> list[pd.Timestamp]:
    """Periods of length `freq` between start and end that contain no time value.

    An empty period inside the data usually means collection stopped, not that
    behaviour stopped. Start and end default to the first and last time value.
    """
    t = pd.to_datetime(pd.Series(np.asarray(times))).dt.floor(freq)
    seen = set(t.unique())
    lo = pd.Timestamp(start).floor(freq) if start is not None else t.min()
    hi = pd.Timestamp(end).floor(freq) if end is not None else t.max()
    return [p for p in pd.date_range(lo, hi, freq=freq) if p not in seen]


def observable(starts, horizon, data_end, empty=()) -> pd.Series:
    """Why each row's outcome cannot be observed: '' (observable), 'censored' or 'empty'.

    A row is censored when start + horizon falls after the last observed period,
    and empty when start + horizon falls on a period with no data. Neither may
    be labelled negative; both should be dropped.
    """
    s = pd.to_datetime(pd.Series(np.asarray(starts)))
    target = s + pd.Timedelta(horizon)
    censored = target > pd.Timestamp(data_end)
    on_empty = target.isin(set(pd.to_datetime(list(empty)))) & ~censored
    return pd.Series("", index=s.index).mask(censored, "censored").mask(on_empty, "empty")
