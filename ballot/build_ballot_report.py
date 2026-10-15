"""Match the judges on a Cook County ballot to their sentencing records.

    python -m ballot.build_ballot_report                         # 2026 retention judges
    python -m ballot.build_ballot_report --ballot path/to.csv    # any CSV with name/assignment columns
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ballot.adapters import cook_bar_ratings, cook_county, il_campaign_finance
from ballot.build_judge_profiles import OUT, RAW, format_profile
from ballot.judges import match, metrics, profile
from ballot.judges.extract import output_path

DEFAULT_BALLOT = OUT.parent / "ballot" / "cook_2026_retention.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ballot", default=DEFAULT_BALLOT)
    parser.add_argument("--since", default="2019-01-01")
    parser.add_argument("--extractions", default="llama", help="whose ISBA extractions to use (llama or claude)")
    args = parser.parse_args()

    ballot = pd.read_csv(args.ballot).fillna("")
    ratings = cook_bar_ratings.load(RAW.parent / "bar_ratings")
    ratings.to_csv(OUT.parent / "ballot" / f"{Path(args.ballot).stem}_bar_ratings.csv", index=False)
    df = cook_county.load(cook_county.fetch(RAW))
    record_names = df.judge.unique()

    # Relabel matching records with the ballot spelling so variants are pooled.
    relabel, matched = {}, {}
    for name in ballot.name:
        matched[name] = match.find_records(name, record_names)
        relabel.update({r: name for r in matched[name]})
    df["judge"] = df.judge.replace(relabel)

    recent = df[df.sentence_date >= args.since].judge.value_counts()
    profiles = {
        p["judge"]: p
        for p in metrics.all_profiles(df, since=args.since, judges=ballot.name, min_cases=metrics.MIN_GROUP)
    }

    isba = ratings[ratings.source == "Illinois State Bar Association"]
    extraction_file = output_path(args.extractions)
    extractions = ({j["name"]: j["extraction"] for j in json.loads(extraction_file.read_text())["judges"]}
                   if extraction_file.exists() else {})
    data_through = df.sentence_date.quantile(0.999).date().isoformat()

    # Campaign money: needs `python -m ballot.build_campaign_finance` to have fetched the receipts.
    isbe = RAW.parent / "isbe"
    receipts = None
    if (isbe / "judge_receipts.csv").exists() and (isbe / "judge_committees.csv").exists():
        committees = pd.read_csv(isbe / "judge_committees.csv", dtype=str)
        receipts = il_campaign_finance.load_receipts(isbe / "judge_receipts.csv", committees)

    report = []
    for row in ballot.itertuples(index=False):
        n = int(recent.get(row.name, 0))
        status = "profile" if n >= metrics.MIN_CASES else "limited" if n >= metrics.MIN_GROUP else "no record"
        judge_ratings = ratings[[match.matches(row.name, r) for r in ratings.name]]
        ballot_numbers = judge_ratings.ballot_number.dropna()
        isba_row = isba[[match.matches(row.name, r) for r in isba.name]]
        isba_name = isba_row.name.iloc[0] if len(isba_row) else None
        report.append({**row._asdict(),
                       "ballot_number": int(ballot_numbers.iloc[0]) if len(ballot_numbers) else None,
                       "record_names": matched[row.name], "n_sentences": n, "status": status,
                       "bar_ratings": [{k: v for k, v in r.items() if pd.notna(v) and k != "summary"}
                                       for r in judge_ratings.to_dict("records")],
                       "bar_summary": bar_summary(judge_ratings),
                       **profile.background_section(isba_row.summary.iloc[0] if isba_name else None,
                                                    extractions.get(isba_name), n),
                       "sentencing": profile.sentencing_section(n, profiles.get(row.name), args.since, data_through),
                       "campaign": (il_campaign_finance.judge_summary(receipts, row.name)
                                    if receipts is not None and row.name in set(receipts.judge) else None)})

    # Every judge with a full record, plus ballot judges with a limited one, for the comparison chart.
    numbers = {r["name"]: r["ballot_number"] for r in report}
    population = metrics.all_profiles(df, since=args.since)
    population += [p for name, p in profiles.items() if name not in {q["judge"] for q in population}]
    population = [profile.population_point(p, numbers.get(p["judge"])) for p in population]

    out = OUT / f"{Path(args.ballot).stem}_judges.json"
    meta = {"source": cook_county.SOURCE_URL, "since": args.since, "data_through": data_through,
            "isba_extractions": str(extraction_file.name) if extractions else None}
    out.write_text(json.dumps({**meta, "judges": report, "population": population}, indent=2, default=float))

    table = pd.DataFrame(report)[["ballot_number", "name", "assignment", "n_sentences", "status", "bar_summary"]]
    table["ballot_number"] = table.ballot_number.astype("Int64")
    table["criticized"] = [bool(r["attorney_criticism"]) for r in report]
    table["background"] = ["; ".join(b["label"].split(" (")[0] for b in r["background"]) for r in report]
    print(table.sort_values("ballot_number").to_string(index=False))
    print(f"\n{(table.status == 'profile').sum()} full profiles, {(table.status == 'limited').sum()} limited, "
          f"{(table.status == 'no record').sum()} without a sentencing record since {args.since} -> {out}")
    for r in sorted(report, key=lambda r: -r["n_sentences"]):
        if "details" in r["sentencing"]:
            print("\n" + format_profile(r["sentencing"]["details"]))


def bar_summary(judge_ratings: pd.DataFrame) -> str:
    """One line per judge, e.g. "Alliance: all yes | ISBA: Yes | CBA: Qualified"."""
    if judge_ratings.empty:
        return "not found in any bar rating source"
    short = {"Alliance of Bar Associations": "Alliance", "Illinois State Bar Association": "ISBA",
             "Chicago Bar Association": "CBA"}
    return " | ".join(f"{short[r.source]}: {r.rating}" for r in judge_ratings.itertuples())


if __name__ == "__main__":
    main()
