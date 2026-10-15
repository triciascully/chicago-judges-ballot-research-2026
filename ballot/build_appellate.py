"""Collect Illinois Appellate Court decisions reviewing each judge on the ballot.

    python -m ballot.build_appellate        # all judges; resumes from cached opinions

Writes data/processed/appellate_decisions.json (one row per confirmed decision).
"""
import json

import pandas as pd

from ballot.adapters import courtlistener
from ballot.build_ballot_report import DEFAULT_BALLOT
from ballot.build_judge_profiles import OUT

OUTFILE = OUT / "appellate_decisions.json"


def main() -> None:
    rows = json.loads(OUTFILE.read_text()) if OUTFILE.exists() else []
    finished = {r["judge"] for r in rows} | set(json.loads((OUT / "appellate_done.json").read_text())
                                                if (OUT / "appellate_done.json").exists() else [])
    for name in pd.read_csv(DEFAULT_BALLOT).name:
        if name in finished:
            continue
        decisions = courtlistener.judge_decisions(name)
        rows += decisions
        finished.add(name)
        OUTFILE.write_text(json.dumps(rows, indent=1))
        (OUT / "appellate_done.json").write_text(json.dumps(sorted(finished)))
        counts = pd.Series([d["outcome"] or "unclear" for d in decisions]).value_counts().to_dict()
        print(f"{name}: {len(decisions)} decisions {counts}", flush=True)
    print(f"{len(rows)} decisions for {len({r['judge'] for r in rows})} judges -> {OUTFILE}")


if __name__ == "__main__":
    main()
