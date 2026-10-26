import pytest

from f1 import ingest

# tests that read the committed data store (data/results, data/openf1, data/analytics) are marked
# `data`; `pytest -m "not data"` runs the self-contained unit tests only
DATA_FIXTURES = {"results", "tables"}
DATA_MODULES = {"test_leakage", "test_analytics"}


def pytest_collection_modifyitems(items):
    for item in items:
        uses_data = DATA_FIXTURES & set(getattr(item, "fixturenames", ()))
        synthetic = item.get_closest_marker("synthetic")
        if (uses_data or item.module.__name__.rsplit(".", 1)[-1] in DATA_MODULES) and not synthetic:
            item.add_marker(pytest.mark.data)


@pytest.fixture(scope="session")
def results():
    return ingest.load_results(seasons=range(2019, 2027))
