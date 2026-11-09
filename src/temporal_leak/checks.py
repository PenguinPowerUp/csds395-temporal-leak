"""Checks on a split, on feature times, and on which rows a fitted step saw."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .errors import LeakageError

DEFAULT_ORDER = ("train", "validation", "test")


def check_temporal_split(times, parts, order: Sequence[str] = DEFAULT_ORDER) -> dict:
    """Every time value sits in one part, and the parts follow each other in time.

    `times` holds one time value per row (the value the split is made on, e.g. a
    cohort date); `parts` holds each row's part name. Returns the first and last
    time and the row count of each part.
    """
    df = pd.DataFrame({"t": np.asarray(times), "p": np.asarray(parts)})
    unknown = sorted(set(df["p"].unique()) - set(order))
    if unknown:
        raise LeakageError(f"unknown part names {unknown}; expected {list(order)}")
    shared = df.groupby("t")["p"].nunique()
    shared = shared[shared > 1]
    if len(shared):
        raise LeakageError(f"{len(shared)} time values appear in more than one part, e.g. {list(shared.index[:3])}")
    ranges = {p: (df.loc[df["p"] == p, "t"].min(), df.loc[df["p"] == p, "t"].max(), int((df["p"] == p).sum()))
              for p in order if (df["p"] == p).any()}
    present = list(ranges)
    for a, b in zip(present, present[1:]):
        if not ranges[a][1] < ranges[b][0]:
            raise LeakageError(f"part '{a}' ends at {ranges[a][1]} but '{b}' starts at {ranges[b][0]}")
    return {p: {"first": r[0], "last": r[1], "rows": r[2]} for p, r in ranges.items()}


def check_before_cutoff(event_times, cutoffs, strict: bool = True) -> int:
    """Every event used to build a row's features happened before that row's scoring instant.

    `event_times` and `cutoffs` are aligned. Missing event times (no event) are
    ignored. With strict=False an event exactly at the cutoff is allowed, which
    suits quantities clipped at the cutoff. Returns the number of events checked.
    """
    e = pd.to_datetime(pd.Series(np.asarray(event_times)))
    c = pd.to_datetime(pd.Series(np.asarray(cutoffs)))
    if len(e) != len(c):
        raise ValueError("event_times and cutoffs must have the same length")
    known = e.notna()
    late = known & ((e >= c) if strict else (e > c))
    if late.any():
        i = int(np.flatnonzero(late.to_numpy())[0])
        raise LeakageError(f"{int(late.sum())} events at or after their cutoff, e.g. event {e.iloc[i]} with cutoff {c.iloc[i]}")
    return int(known.sum())


def check_fit_rows(fit_rows, allowed_rows) -> int:
    """A fitted step (scaler, calibrator, threshold) saw only allowed rows.

    Both arguments are boolean masks of the same length. Returns the number of
    rows the step was fit on.
    """
    f = np.asarray(fit_rows, dtype=bool)
    a = np.asarray(allowed_rows, dtype=bool)
    if f.shape != a.shape:
        raise ValueError("fit_rows and allowed_rows must have the same shape")
    bad = f & ~a
    if bad.any():
        raise LeakageError(f"{int(bad.sum())} rows used for fitting are not allowed, e.g. row {int(np.flatnonzero(bad)[0])}")
    return int(f.sum())
