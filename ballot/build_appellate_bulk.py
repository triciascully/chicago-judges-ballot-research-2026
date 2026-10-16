"""Find every Illinois Appellate Court decision that reviewed a judge on the ballot, from the
bulk-data Parquet files (see ballot.adapters.courtlistener_bulk). No API calls, no limits.

    python -m ballot.build_appellate_bulk

Each decision's header names the trial judge ("Appeal from the Circuit Court of Cook County ...
The Honorable Jane Q. Doe, Judge, presiding."); the same caption check and disposition reader
as the API pipeline decide which judge it reviewed and how it came out.
Writes data/processed/appellate_decisions.json (one row per decision) and a Parquet copy.
"""
import json
import re

import pandas as pd

from ballot.adapters import courtlistener
from ballot.adapters.courtlistener_bulk import OUT as BULK_OUT
from ballot.build_ballot_report import DEFAULT_BALLOT
from ballot.build_judge_profiles import OUT

DECISIONS = OUT / "appellate_decisions.json"
# The opinion that decides the case: CourtListener's combined or lead opinion, before concurrences.
_MAIN_TYPES = ["010combined", "020lead", "015unamimous", "025plurality", "030concurrence"]


def load() -> pd.DataFrame:
    clusters = pd.read_parquet(BULK_OUT / "clusters.parquet", filters=[("court_id", "==", "illappct")])
    ops = pd.read_parquet(BULK_OUT / "opinions.parquet")
    ops = ops[ops.cluster_id.isin(clusters.id)]
    dockets = pd.read_parquet(BULK_OUT / "dockets.parquet", columns=["id", "docket_number"])
    ops["rank"] = ops["type"].map({t: i for i, t in enumerate(_MAIN_TYPES)}).fillna(len(_MAIN_TYPES))
    ops = ops.sort_values("rank").drop_duplicates("cluster_id")
    df = ops.merge(clusters, left_on="cluster_id", right_on="id", suffixes=("", "_cluster"))
    return df.merge(dockets, left_on="docket_id", right_on="id", suffixes=("", "_docket"))


def main() -> None:
    names = pd.read_csv(DEFAULT_BALLOT).name.tolist()
    df = load()
    print(f"{len(df):,} Illinois Appellate Court decisions since {df.date_filed.min()}")
    head = df.text.str.slice(0, 8000)
    cook = head.str.contains("Cook County", case=False) & head.str.contains("presiding", case=False)
    print(f"{cook.sum():,} are appeals from Cook County with a presiding judge named")

    rows = []
    for r in df[cook].itertuples():
        judge = next((n for n in names if courtlistener.presided(r.text, n)), None)
        if judge is None:
            continue
        rows.append({
            "judge": judge, "case_name": r.case_name, "date": r.date_filed, "docket": r.docket_number,
            "status": r.precedential_status, "opinion_id": r.id, "cluster_id": r.cluster_id,
            "url": f"https://www.courtlistener.com/opinion/{r.cluster_id}/",
            "outcome": courtlistener.outcome(r.text),
            "disposition_field": (r.disposition or "")[:200],
            "criminal": bool(re.match(r"People v\.?", r.case_name or "")),
        })
    out = pd.DataFrame(rows).sort_values(["judge", "date"])
    DECISIONS.write_text(json.dumps(out.to_dict("records"), indent=1))
    out.to_parquet(OUT / "appellate_decisions.parquet", index=False)

    summary = out.assign(outcome=out.outcome.fillna("unclear")).groupby("judge").outcome.value_counts().unstack(fill_value=0)
    print(summary.to_string())
    print(f"\n{len(out):,} decisions for {out.judge.nunique()} of {len(names)} ballot judges "
          f"({out.outcome.isna().sum()} with no clear outcome) -> {DECISIONS}")


if __name__ == "__main__":
    main()
