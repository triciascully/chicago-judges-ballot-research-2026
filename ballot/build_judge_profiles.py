"""Build sentencing profiles for Cook County judges.

    python -m ballot.build_judge_profiles                  # all judges -> data/processed/
    python -m ballot.build_judge_profiles --judge "Linn"   # also print matching profiles
"""
import argparse
import json
from pathlib import Path

from ballot.adapters import cook_county
from ballot.judges import metrics

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "cook_sentencing.csv"
OUT = ROOT / "data" / "processed"


def _pp(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def format_profile(p: dict) -> str:
    pr, tr = p["prison"], p["treatment"]
    lines = [
        f"{p['judge']}  ({p['n_cases']:,} sentences, {p['first_sentence']} to {p['last_sentence']}, "
        f"{p['felony_share']:.0%} felonies)",
        f"  Prison:     {pr['rate']:.0%} of cases vs {pr['expected']:.0%} expected for comparable charges "
        f"({_pp(pr['diff'])}, 95% CI {_pp(pr['lo'])} to {_pp(pr['hi'])})",
        f"  Treatment:  {tr['rate']:.1%} vs {tr['expected']:.1%} expected "
        f"({_pp(tr['diff'])}, 95% CI {_pp(tr['lo'])} to {_pp(tr['hi'])})",
    ]
    if p["n_cases"] < metrics.MIN_CASES:
        lines.insert(1, "  Note: limited record; treat these numbers as rough.")
    if p["likely_specialty_court"]:
        lines.insert(1, "  Note: mostly treatment sentences; likely a specialty court, so comparisons are limited.")
    if ln := p.get("prison_length_ratio"):
        lines.append(f"  Prison terms: {ln['ratio']:.2f}x comparable terms (95% CI {ln['lo']:.2f}-{ln['hi']:.2f}, "
                     f"n={ln['n']:,})")
    for race, r in p["prison_by_race"].items():
        lines.append(f"    {race:<7} prison {r['rate']:.0%} vs {r['expected']:.0%} expected "
                     f"({_pp(r['diff'])}, n={r['n']:,})")
    if gap := p.get("black_white_prison_gap"):
        lines.append(f"  Black-white prison gap, comparable charges: {_pp(gap['gap'])} "
                     f"(95% CI {_pp(gap['lo'])} to {_pp(gap['hi'])})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2019-01-01", help="only use sentences on or after this date")
    parser.add_argument("--judge", help="print profiles for judges whose name contains this text")
    parser.add_argument("--refresh", action="store_true", help="re-download the source data")
    args = parser.parse_args()

    df = cook_county.load(cook_county.fetch(RAW, refresh=args.refresh))
    profiles = metrics.all_profiles(df, since=args.since)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics.summary_table(profiles).to_csv(OUT / "cook_judges.csv", index=False)
    meta = {"jurisdiction": cook_county.JURISDICTION, "source": cook_county.SOURCE_URL, "since": args.since}
    (OUT / "cook_judges.json").write_text(json.dumps({**meta, "judges": profiles}, indent=2, default=float))
    print(f"{len(profiles)} judges with >= {metrics.MIN_CASES} sentences since {args.since} -> {OUT}")

    if args.judge:
        for p in profiles:
            if args.judge.lower() in p["judge"].lower():
                print("\n" + format_profile(p))


if __name__ == "__main__":
    main()
