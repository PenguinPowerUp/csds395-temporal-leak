# temporal-leak

Small checks that a time-indexed prediction pipeline never uses information from after its scoring instant.

## Install

```bash
pip install "temporal-leak @ git+https://github.com/PenguinPowerUp/csds395-temporal-leak@v0.1.0"
```

## Checks

| Function                                                                                                       | What it catches                                                                                                                |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `check_temporal_split(times, parts)`                                                                         | A time value in two parts, or parts out of time order (train must end before validation starts, validation before test)        |
| `check_before_cutoff(event_times, cutoffs)`                                                                  | A feature built from an event at or after the row's scoring instant                                                            |
| `check_fit_rows(fit_rows, allowed_rows)`                                                                     | A scaler, calibrator or threshold fit on rows it must not see, such as test rows                                               |
| `empty_periods(times, freq)`                                                                                 | Periods with no data at all, which usually mean collection stopped                                                             |
| `observable(starts, horizon, data_end, empty)`                                                               | Rows whose outcome window lies past the data or on an empty period; drop them rather than label them negative                  |
| `random_split(n, fractions, seed)` and `shuffle_test(run, temporal_parts, random_parts, metric, max_gain)` | Runs the same pipeline on a temporal and a random split; a random split that scores much higher reveals time-local information |

## Example

```python
import pandas as pd
import temporal_leak as tl

tl.check_temporal_split(df["cohort_date"], df["part"])
tl.check_before_cutoff(df["last_event_time"], df["cohort_date"] + pd.Timedelta(days=1))
tl.check_fit_rows(scaler_rows, df["part"] == "train")
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

retention-bench (https://github.com/PenguinPowerUp/csds395-retention-bench) uses every check on two public game datasets.
