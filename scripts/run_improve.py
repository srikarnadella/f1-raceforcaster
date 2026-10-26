"""Run one improvement cycle manually. Usage: python scripts/run_improve.py [--dry-run]"""
import json
import sys


from f1 import improve, ingest

cal = ingest.load_calendar()
r = ingest.load_results()
last = r[["season", "round"]].drop_duplicates().iloc[-1]
nxt = cal[(cal.season == last.season) & (cal["round"] > last["round"])]
key = (int(last.season), int(nxt["round"].iloc[0])) if len(nxt) else (int(last.season) + 1, 1)
print(json.dumps(improve.run_cycle(key, dry_run="--dry-run" in sys.argv), indent=1, default=float))
