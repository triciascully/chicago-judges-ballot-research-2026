"""Does matching on geography change how ballot judges compare? A Cook County lawyer pointed
out that courtrooms take cases from fixed police districts, so judges see different caseloads.

    python -m ballot.judges.geography_check

Reruns the prison-rate comparison three ways for every ballot judge with a profile:
  charge      same charge class and offense type, countywide (the original method)
  courthouse  same charge at the same courthouse (what the guide now uses)
  district    same charge from the same Chicago police district, on only the cases that
              record a district (about a quarter of Chicago arrests), so samples are small
and prints how much each judge's gap from expected moves, plus whether its finding flips.
"""
import pandas as pd

from ballot.adapters import cook_county
from ballot.build_ballot_report import DEFAULT_BALLOT, RAW
from ballot.judges import match, metrics

METHODS = {
    "charge": metrics.STRATA[1:],
    "courthouse": metrics.STRATA,
    "district": [["police_district", "offense_class", "offense_category"], ["police_district", "offense_class"]],
}


def finding(p: dict) -> str:
    return "above" if p["lo"] > 0 else "below" if p["hi"] < 0 else "none"


def main(since: str = "2019-01-01") -> None:
    df = cook_county.load(cook_county.fetch(RAW))
    names = pd.read_csv(DEFAULT_BALLOT).name
    record_names = df.judge.dropna().unique()
    df["judge"] = df.judge.replace({r: n for n in names for r in match.find_records(n, record_names)})
    rows = {}
    for method, strata in METHODS.items():
        data = df[df.police_district.notna()] if method == "district" else df
        for p in metrics.all_profiles(data, since, judges=names, min_cases=metrics.MIN_GROUP, strata=strata):
            r = rows.setdefault(p["judge"], {"judge": p["judge"]})
            r[f"{method}_n"] = p["n_cases"]
            r[f"{method}_pp"] = round(100 * p["prison"]["diff"], 1)
            r[f"{method}_finding"] = finding(p["prison"])
    out = pd.DataFrame(rows.values()).sort_values("judge")
    print(out.to_string(index=False))
    both = out.dropna(subset=["charge_pp", "courthouse_pp"])
    print(f"\ncharge -> courthouse: median shift {(both.courthouse_pp - both.charge_pp).abs().median():.1f} pts, "
          f"findings changed for {(both.charge_finding != both.courthouse_finding).sum()} of {len(both)} judges")
    d = out.dropna(subset=["district_pp"])
    d = d[d.district_n >= metrics.MIN_CASES]
    if len(d):
        print(f"district check ({len(d)} judges with {metrics.MIN_CASES}+ district-coded cases): "
              f"median shift vs charge {(d.district_pp - d.charge_pp).abs().median():.1f} pts, "
              f"findings changed for {(d.charge_finding != d.district_finding).sum()}")


if __name__ == "__main__":
    main()
