"""Gate for the scheduled workflow: prints which job is due and writes it to
$GITHUB_OUTPUT (forecast=<season>_<round>, review=..., overdue=...)."""
import json
import os
import sys

import pandas as pd

from f1 import ingest, state


def main():
    year = pd.Timestamp.utcnow().year
    try:
        ingest.update(seasons=[year], current=year)  # refresh calendar and current season
    except Exception as ex:
        print(f"refresh failed ({ex}); using committed calendar", file=sys.stderr)
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
