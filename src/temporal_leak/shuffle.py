"""The shuffle test: the same model on a temporal split and on a random split."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .errors import LeakageError

PART_NAMES = ("train", "validation", "test")


def random_split(n: int, fractions: tuple[float, float], seed: int, names: Sequence[str] = PART_NAMES) -> np.ndarray:
    """Assign n rows to three parts uniformly at random, with the given cumulative fractions."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    part = np.empty(n, dtype=object)
    a, b = int(fractions[0] * n), int(fractions[1] * n)
    part[order[:a]] = names[0]
    part[order[a:b]] = names[1]
    part[order[b:]] = names[2]
    return part


def shuffle_test(run: Callable, temporal_parts, random_parts, metric: Callable | None = None,
                 max_gain: float | None = None) -> dict:
    """Run the same pipeline on both splits and compare them.

    `run(parts)` fits and evaluates the pipeline on one assignment of rows to
    parts and returns anything. If `metric(result)` is given, the gain of the
    random split over the temporal split is reported, and with `max_gain` a
    larger gain raises LeakageError: a random split that scores much higher
    reveals time-local information that the temporal split correctly withholds.
    """
    out = {"temporal": run(temporal_parts), "random": run(random_parts)}
    if metric is not None:
        out["gain"] = float(metric(out["random"]) - metric(out["temporal"]))
        if max_gain is not None and out["gain"] > max_gain:
            raise LeakageError(f"random split beats temporal split by {out['gain']:.4f} > {max_gain}")
    return out
