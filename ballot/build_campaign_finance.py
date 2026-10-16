"""Fetch campaign receipts for the judges on the ballot (about 10 minutes: it streams a 1 GB file).

    python -m ballot.build_campaign_finance
    python -m ballot.build_ballot_report      # then rebuild profiles to include campaign money
"""
import pandas as pd

from ballot.adapters import il_campaign_finance
from ballot.build_ballot_report import DEFAULT_BALLOT
from ballot.build_judge_profiles import RAW

SINCE = "2020-01-01"  # the current judges' last retention campaign through now


def main() -> None:
    isbe = RAW.parent / "isbe"
    il_campaign_finance.fetch_lookups(isbe, refresh=True)
    committees = il_campaign_finance.judge_committees(isbe, pd.read_csv(DEFAULT_BALLOT).name)
    committees.to_csv(isbe / "judge_committees.csv", index=False)
    print(f"{len(committees)} committees for {committees.judge.nunique()} judges; streaming receipts...")
    path = il_campaign_finance.fetch_receipts(set(committees.committee_id), isbe / "judge_receipts.csv", SINCE)
    print(f"{sum(1 for _ in path.open()) - 1} receipts since {SINCE} -> {path}")

    # Where the judges' organizational donors give elsewhere (a second pass over Receipts.txt).
    receipts = il_campaign_finance.load_receipts(path, committees)
    names = il_campaign_finance.donor_org_names(receipts)
    giving = il_campaign_finance.fetch_org_giving(names, isbe / "org_giving.csv", since="2016-01-01")
    print(f"giving by {len(names)} donor organizations since 2016 -> {giving}")


if __name__ == "__main__":
    main()
