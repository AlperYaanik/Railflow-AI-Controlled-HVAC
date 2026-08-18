"""Session-wide pytest setup.

Forces lightgbm to be the first of {lightgbm, pandas} imported into the test
process. On this Windows environment, if pandas' native runtime loads first,
every subsequent lightgbm.Dataset call dies with an access violation deep in
LGBM_DatasetSetField -- a hard native crash, not a catchable Python exception.
Confirmed with a minimal repro unrelated to this project's data: `import
pandas` (never used) followed by `import lightgbm` and a single Dataset
construct() call on a random array crashes the same way; reversing the import
order avoids it completely. See src/train.py for the full note.

conftest.py is imported by pytest before it collects and imports test modules
in this directory, so this is the one place that can guarantee the ordering
regardless of which test file pytest happens to import first -- reordering
imports inside src/train.py alone is not enough, since test_data_generator.py
and test_features.py both import pandas and could easily run first.
"""

import lightgbm  # noqa: F401  (import order side effect; not used directly here)

import pytest


@pytest.fixture(scope="session")
def require_weather():
    """Skip with an actionable message if the weather cache is missing.

    Every M4 test that generates scenarios needs weather_{city}_summer.csv,
    which is gitignored (regenerable, not committed -- same pattern as the
    M1 weather cache). A fresh clone has none of it. Without this fixture,
    tests that call generate_dataset()/train_model() directly (rather than
    through an already-guarded fixture) fail with a raw FileNotFoundError
    from deep inside pandas instead of saying what to do about it.
    """
    from src.config import DATA_DIR

    missing = [f"weather_{city}_summer.csv" for city in ("cairo", "aswan")
               if not (DATA_DIR / f"weather_{city}_summer.csv").exists()]
    if missing:
        pytest.skip(f"weather cache missing {missing}. Run:  python -m src.weather")
