"""Gate for the scheduled workflow: prints which job is due and writes it to
$GITHUB_OUTPUT (forecast=<season>_<round>, review=..., overdue=...)."""
import json
import os
import sys

import pandas as pd

from f1 import championship, ingest, review, state


def main():
    year = pd.Timestamp.utcnow().year
    try:
        ingest.update(seasons=[year], current=year)  # refresh calendar and current season
    except Exception as ex:
        print(f"refresh failed ({ex}); using committed calendar", file=sys.stderr)
    results = ingest.load_results()
    redone = championship.refresh_stale(results, ingest.load_calendar(), year)
    for rnd in redone:  # the weekend review's over/under and misses must use the corrected result too
        review.pace_report(results, year, rnd)
    if redone:
        print(f"championship re-simulated after result corrections: rounds {redone}", file=sys.stderr)
    due = state.due_jobs(ingest.load_calendar())
    out = {"forecast": state.key(*due["forecast"]) if due["forecast"] else "",
           "review": state.key(*due["review"]) if due["review"] else "",
           "overdue": ",".join(due["overdue"])}
    print(json.dumps(out))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for k, v in out.items():
                f.write(f"{k}={v}\n")


if __name__ == "__main__":
    main()
