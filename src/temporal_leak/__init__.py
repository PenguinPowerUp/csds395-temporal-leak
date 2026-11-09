"""temporal-leak: checks that a time-indexed prediction pipeline never uses the future."""

from .checks import check_before_cutoff, check_fit_rows, check_temporal_split
from .errors import LeakageError
from .periods import empty_periods, observable
from .shuffle import random_split, shuffle_test
