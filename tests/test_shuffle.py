import numpy as np
import pytest

from temporal_leak import LeakageError, random_split, shuffle_test


def test_random_split_sizes_and_reproducibility():
    a = random_split(100, (0.6, 0.8), seed=1)
    assert (a == "train").sum() == 60 and (a == "validation").sum() == 20 and (a == "test").sum() == 20
    assert (a == random_split(100, (0.6, 0.8), seed=1)).all()


def test_shuffle_test_reports_gain_and_flags_a_large_one():
    temporal = np.array(["train"] * 5 + ["test"] * 5, dtype=object)
    shuffled = random_split(10, (0.5, 0.5), seed=0)
    scores = {id(temporal): 0.60, id(shuffled): 0.70}
    run = lambda parts: scores[id(parts)]  # noqa: E731
    out = shuffle_test(run, temporal, shuffled, metric=lambda s: s)
    assert out["gain"] == pytest.approx(0.10)
    with pytest.raises(LeakageError, match="beats"):
        shuffle_test(run, temporal, shuffled, metric=lambda s: s, max_gain=0.05)
