import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # fast and identical on a shared CI runner

import pytest  # noqa: E402

from crash_whatif import study as st  # noqa: E402
from crash_whatif.config import Settings  # noqa: E402
from crash_whatif.synthetic import make_crashes  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return Settings(n_queries=10)


@pytest.fixture(scope="session")
def study(settings):
    s = st.prepare(make_crashes(4000, seed=3), settings)
    return st.fit_all(s, names=("majority", "make_only", "logreg", "random_forest"))
