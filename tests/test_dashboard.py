"""Smoke test: every dashboard page renders without an exception (CI gate before deploy)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [("analytics_pages", p) for p in ("projections", "weekend_lab", "race_story", "team_dna", "driver_dna",
                                          "circuits", "ml_lab")] + \
        [("explore_pages", p) for p in ("driver_page", "constructor_page", "track_page", "eras_page", "season_page")] + \
        [("unsupervised_page", "unsupervised_page")]


@pytest.mark.data
@pytest.mark.parametrize("module, page", PAGES)
def test_page_renders(module, page):
    at = AppTest.from_string(f"import sys; sys.path.insert(0, {str(APP)!r})\nimport ui; ui.setup()\n"
                             f"import {module} as M\nM.{page}()\n", default_timeout=240).run()
    assert not at.exception, at.exception[0].message
