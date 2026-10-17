"""Match the judges on a Cook County ballot to their sentencing records.

    python -m ballot.build_ballot_report                         # 2026 retention judges
    python -m ballot.build_ballot_report --ballot path/to.csv    # any CSV with name/assignment columns
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ballot.adapters import cook_bar_ratings, cook_county, il_campaign_finance
from ballot.build_judge_profiles import OUT, RAW, format_profile
from ballot.judges import match, metrics, profile
from ballot.judges.extract import output_path

DEFAULT_BALLOT = OUT.parent / "ballot" / "cook_2026_retention.csv"
OPEN_SEATS = OUT.parent / "ballot" / "cook_2026_open_seats.csv"


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
    # Candidates for open seats (contested races). Some are already judges with a record.
    open_seats = pd.read_csv(OPEN_SEATS).fillna("") if Path(OPEN_SEATS).exists() else pd.DataFrame(columns=["name"])
    contested_ratings = cook_bar_ratings.load_contested(RAW.parent / "bar_ratings") if len(open_seats) else ratings.iloc[:0]

    # Relabel matching records with the ballot spelling so variants are pooled.
    relabel, matched = {}, {}
    for name in pd.concat([ballot.name, open_seats.name]).drop_duplicates():
        matched[name] = match.find_records(name, record_names)
        relabel.update({r: name for r in matched[name]})
    df["judge"] = df.judge.replace(relabel)

    recent = df[df.sentence_date >= args.since].judge.value_counts()
    profiles = {
        p["judge"]: p
        for p in metrics.all_profiles(df, since=args.since, judges=list(ballot.name) + list(open_seats.name),
                                      min_cases=metrics.MIN_GROUP)
    }

    isba = ratings[ratings.source == "Illinois State Bar Association"]
    extraction_file = output_path(args.extractions)
    extractions = ({j["name"]: j["extraction"] for j in json.loads(extraction_file.read_text())["judges"]}
                   if extraction_file.exists() else {})
    data_through = df.sentence_date.quantile(0.999).date().isoformat()

    # Judges also running for another seat this year. Their campaign committees for that race are
    # still their money, so they stay in the totals; this note says where it came from.
    also_running = {
        "Judith C. Rice": "Also on the November ballot, unopposed, for the Illinois Appellate Court, First District "
                          "(won the March 2026 primary). Her committee for that race, Citizens for Judge Judith C. Rice "
                          "(opened August 2025), is included in these totals.",
    }

    # Injustice Watch's guide pages (their public sitemap). Where two pages share a name, the
    # retention judge's has the higher id: the guide numbered retention pages from 35786 up.
    iw_pages = []
    if (OUT.parent / "ballot" / "injustice_watch_2026_pages.txt").exists():
        iw_pages = (OUT.parent / "ballot" / "injustice_watch_2026_pages.txt").read_text().split()

    def iw_url(name: str) -> str | None:
        slug = lambda u: re.sub(r"-\d+/?$", "", u.rstrip("/").split("/")[-1]).replace("-", " ")
        hits = [u for u in iw_pages if match.matches(name, slug(u))]
        if not hits:
            return None
        best = max(hits, key=lambda u: int(re.search(r"-(\d+)/?$", u).group(1)))
        return best.replace(".org/", ".org/candidate/", 1)

    # Appellate track record (from ballot.build_appellate_bulk).
    appeals = json.loads((OUT / "appellate_decisions.json").read_text()) if (OUT / "appellate_decisions.json").exists() else []
    appeal_baseline = json.loads((OUT / "appellate_baseline.json").read_text()) if (OUT / "appellate_baseline.json").exists() else None

    # Campaign money: needs `python -m ballot.build_campaign_finance` to have fetched the receipts.
    isbe = RAW.parent / "isbe"
    receipts = None
    if (isbe / "judge_receipts.csv").exists() and (isbe / "judge_committees.csv").exists():
        committees = pd.read_csv(isbe / "judge_committees.csv", dtype=str)
        receipts = il_campaign_finance.load_receipts(isbe / "judge_receipts.csv", committees)
    parties = (il_campaign_finance.judge_party(isbe, pd.concat([ballot.name, open_seats.name]))
               if (isbe / "Candidates.txt").exists() else {})
    # Where the judges' organizational donors give elsewhere (needs org_giving.csv; see
    # ballot.build_campaign_finance). The judges' own committees are excluded from that measure.
    leanings = {}
    if receipts is not None and (isbe / "org_giving.csv").exists():
        own = set(committees.committee_id) | il_campaign_finance.JOINT_COMMITTEES
        leanings = il_campaign_finance.org_leanings(isbe / "org_giving.csv",
                                                    il_campaign_finance.committee_parties(isbe), own)

    open_file = output_path(args.extractions, open_seats=True)
    open_extractions = ({j["name"]: j["extraction"] for j in json.loads(open_file.read_text())["judges"]}
                        if open_file.exists() else {})

    def entry(row, rating_rows: pd.DataFrame, isba_rows: pd.DataFrame, extracted: dict) -> dict:
        n = int(recent.get(row.name, 0))
        status = "profile" if n >= metrics.MIN_CASES else "limited" if n >= metrics.MIN_GROUP else "no record"
        judge_ratings = rating_rows[[match.matches(row.name, r) for r in rating_rows.name]]
        ballot_numbers = judge_ratings.ballot_number.dropna()
        isba_row = isba_rows[[match.matches(row.name, r) for r in isba_rows.name]]
        isba_name = isba_row.name.iloc[0] if len(isba_row) else None
        return {**row._asdict(),
                "ballot_number": int(ballot_numbers.iloc[0]) if len(ballot_numbers) else None,
                "record_names": matched[row.name], "n_sentences": n, "status": status,
                "bar_ratings": [{k: v for k, v in r.items() if pd.notna(v) and k not in ("summary", "seat", "party")}
                                for r in judge_ratings.to_dict("records")],
                "bar_summary": bar_summary(judge_ratings),
                **profile.background_section(isba_row.summary.iloc[0] if isba_name else None,
                                             extracted.get(isba_name), n),
                "sentencing": profile.sentencing_section(n, profiles.get(row.name), args.since, data_through),
                "party": parties.get(row.name) or getattr(row, "party", ""),
                "injustice_watch_url": iw_url(row.name),
                "also_running": also_running.get(row.name),
                "appellate": profile.appellate_section([a for a in appeals if a["judge"] == row.name],
                                                       appeal_baseline) if appeal_baseline else None,
                "campaign": _with_leanings(il_campaign_finance.judge_summary(receipts, row.name), leanings)
                             if receipts is not None and row.name in set(receipts.judge) else None}

    report = [entry(row, ratings, isba, extractions) for row in ballot.itertuples(index=False)]
    contested_isba = contested_ratings[contested_ratings.source == "Illinois State Bar Association"]
    open_report = []
    for row in open_seats.itertuples(index=False):
        e = entry(row, contested_ratings, contested_isba, open_extractions)
        e["ballot_number"] = int(row.ballot_number)  # the CBA list repeats numbers across subcircuits
        e.pop("appellate")  # the appellate matching covers retention judges only
        open_report.append(e)

    # Every judge with a full record, plus ballot judges with a limited one, for the comparison chart.
    numbers = {r["name"]: r["ballot_number"] for r in report}
    population = metrics.all_profiles(df, since=args.since)
    population += [p for name, p in profiles.items() if name not in {q["judge"] for q in population}]
    population = [profile.population_point(p, numbers.get(p["judge"])) for p in population]

    out = OUT / f"{Path(args.ballot).stem}_judges.json"
    meta = {"source": cook_county.SOURCE_URL, "since": args.since, "data_through": data_through,
            "appellate_baseline": appeal_baseline,
            "isba_extractions": str(extraction_file.name) if extractions else None}
    out.write_text(json.dumps({**meta, "judges": report, "open_seats": open_report, "population": population},
                              indent=2, default=float))

    table = pd.DataFrame(report)[["ballot_number", "name", "assignment", "n_sentences", "status", "bar_summary"]]
    table["ballot_number"] = table.ballot_number.astype("Int64")
    table["criticized"] = [bool(r["attorney_criticism"]) for r in report]
    table["background"] = ["; ".join(b["label"].split(" (")[0] for b in r["background"]) for r in report]
    print(table.sort_values("ballot_number").to_string(index=False))
    print(f"\n{(table.status == 'profile').sum()} full profiles, {(table.status == 'limited').sum()} limited, "
          f"{(table.status == 'no record').sum()} without a sentencing record since {args.since} -> {out}")
    print(f"{len(open_report)} open-seat candidates; with campaign money: {sum(bool(e['campaign']) for e in open_report)}, "
          f"with a sentencing record: {sum(e['status'] != 'no record' for e in open_report)}")
    for r in sorted(report, key=lambda r: -r["n_sentences"]):
        if "details" in r["sentencing"]:
            print("\n" + format_profile(r["sentencing"]["details"]))


def _with_leanings(summary: dict, leanings: dict) -> dict:
    """Attach each listed employer's or organization's other Illinois giving, by party."""
    for e in summary["top_employers"]:
        e["elsewhere"] = leanings.get(e["employer"])
    for o in summary["top_organizations"]:
        o["elsewhere"] = leanings.get(o["name"])
    return summary


def bar_summary(judge_ratings: pd.DataFrame) -> str:
    """One line per judge, e.g. "Alliance: all yes | ISBA: Yes | CBA: Qualified"."""
    if judge_ratings.empty:
        return "not found in any bar rating source"
    short = {"Alliance of Bar Associations": "Alliance", "Illinois State Bar Association": "ISBA",
             "Chicago Bar Association": "CBA"}
    return " | ".join(f"{short[r.source]}: {r.rating}" for r in judge_ratings.itertuples())


if __name__ == "__main__":
    main()
