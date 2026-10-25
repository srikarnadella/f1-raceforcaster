import pytest

from f1 import ingest


@pytest.fixture(scope="session")
def results():
    return ingest.load_results(seasons=range(2019, 2027))
