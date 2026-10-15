"""Collect Illinois Appellate Court decisions reviewing each judge on the ballot.

    python -m ballot.build_appellate        # all judges; resumable

CourtListener search allows 50 requests an hour, so judges are searched in batches and
every search page is cached; a full run takes a few hours and waits out the limit on its own.
Writes data/processed/appellate_decisions.json (one row per confirmed decision).
"""
import json

import pandas as pd

from ballot.adapters import courtlistener
from ballot.build_ballot_report import DEFAULT_BALLOT
from ballot.build_judge_profiles import OUT

OUTFILE = OUT / "appellate_decisions.json"
DONE = OUT / "appellate_done.json"
BATCH = 8  # judges per search query; keeps the query a reasonable length


def main() -> None:
    rows = json.loads(OUTFILE.read_text()) if OUTFILE.exists() else []
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    todo = [n for n in pd.read_csv(DEFAULT_BALLOT).name if n not in done]
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        found = courtlistener.decisions_for(batch)
        rows = [r for r in rows if r["judge"] not in batch] + found
        done |= set(batch)
        OUTFILE.write_text(json.dumps(rows, indent=1))
        DONE.write_text(json.dumps(sorted(done)))
        for name in batch:
            mine = [d for d in found if d["judge"] == name]
            counts = pd.Series([d["outcome"] or "unclear" for d in mine]).value_counts().to_dict()
            print(f"{name}: {len(mine)} decisions {counts}", flush=True)
    print(f"{len(rows)} decisions for {len({r['judge'] for r in rows})} judges -> {OUTFILE}")


if __name__ == "__main__":
    main()
