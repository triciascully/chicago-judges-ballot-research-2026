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
BASELINE = OUT / "appellate_baseline.json"
MERITS = ["affirmed", "reversed", "vacated", "affirmed in part"]  # dismissals never reach the merits


def kind(case_name: str) -> str:
    """Case type, which shapes how often appeals succeed."""
    if re.match(r"People v\.?", case_name or ""):
        return "criminal"
    if re.match(r"In re", case_name or ""):
        return "juvenile/family"
    return "civil"
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

    # Countywide baseline: every Cook County appeal, not only the ballot judges'.
    allcook = df[cook].assign(outcome=[courtlistener.outcome(t) for t in df[cook].text])
    allcook = allcook.sort_values("date_filed").drop_duplicates("docket_number", keep="last")
    allcook["kind"] = allcook.case_name.map(kind)
    merits = allcook[allcook.outcome.isin(MERITS)].assign(changed=lambda d: d.outcome != "affirmed")
    BASELINE.write_text(json.dumps({
        "appeals": len(allcook), "decided_on_merits": len(merits),
        "changed_share": round(float(merits.changed.mean()), 4),
        "by_kind": {k: {"n": int(len(g)), "changed_share": round(float(g.changed.mean()), 4)}
                    for k, g in merits.groupby("kind")},
        "since": str(df.date_filed.min())[:10],
    }, indent=1))

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
            "kind": kind(r.case_name),
        })
    # One appeal can appear twice (an unpublished order later reissued as a published opinion).
    # Illinois gives each appeal its own docket number, so keep the latest version per docket.
    out = (pd.DataFrame(rows).sort_values("date")
           .drop_duplicates(["docket", "judge"], keep="last").sort_values(["judge", "date"]))
    # Missing outcomes must be null, not NaN: NaN isn't valid JSON and breaks the page.
    records = json.loads(out.to_json(orient="records", date_format="iso"))
    DECISIONS.write_text(json.dumps(records, indent=1))
    out.to_parquet(OUT / "appellate_decisions.parquet", index=False)

    summary = out.assign(outcome=out.outcome.fillna("unclear")).groupby("judge").outcome.value_counts().unstack(fill_value=0)
    print(summary.to_string())
    print(f"\n{len(out):,} decisions for {out.judge.nunique()} of {len(names)} ballot judges "
          f"({out.outcome.isna().sum()} with no clear outcome) -> {DECISIONS}")


if __name__ == "__main__":
    main()
