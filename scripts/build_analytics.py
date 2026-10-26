"""Rebuild data/analytics/ from the OpenF1 store (stages in f1.analytics.pipeline).
Deterministic; about a minute.

Usage: python scripts/build_analytics.py [-v]
"""
import json
import sys
import warnings

from f1.analytics.pipeline import _json, run

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    print(json.dumps(run(verbose="-v" in sys.argv), default=_json, indent=1))
