"""How often do a state's appellate opinions name the trial judge? Decides whether a city's
appellate track record can be built the way Cook County's was (Michigan's can't).

    python -m ballot.judges.caption_check            # uses data/processed/courtlistener/cities_*.parquet

For each city, takes appeals from its county since 2020 and looks for that state's usual
caption wording. Prints the share that name a judge, plus examples to eyeball.
"""
import re

import pandas as pd

from ballot.adapters.courtlistener_bulk import OUT

# city: (courts, how appeals from its county are introduced, how the trial judge is named)
CITIES = {
    "Houston": (["texapp", "texcrimapp"], r"Harris County",
                r"(?:Trial Court Judge|Hon(?:orable|\.))[:\s]+([A-Z][A-Za-z.'\- ]{3,40})|([A-Z][A-Za-z.'\- ]{3,40}),\s+Judge Presiding"),
    "Los Angeles": (["calctapp"], r"Superior Court of Los Angeles County",
                    r"Superior Court of Los Angeles County,\s*([A-Z][A-Za-z.'\- ]{3,40}),\s*(?:Judge|Commissioner)"),
    "New York City": (["nyappdiv", "nyappterm"], r"(?:New York|Kings|Queens|Bronx|Richmond) County",
                      r"\(([A-Z][A-Za-z.'\- ]{2,40}),\s*J\.(?:H\.O\.)?\)"),
    "Miami": (["fladistctapp", "fladistctapp3"], r"Miami-Dade County",
              r"Miami-Dade County,\s*([A-Z][A-Za-z.'\- ]{3,40}),\s*Judge"),
    "Milwaukee": (["wisctapp"], r"Milwaukee County",
                  r"Milwaukee County:\s*([A-Z][A-Z.'\- ]{3,40}),\s*(?:Reserve\s+)?Judge"),
}


def main() -> None:
    clusters = pd.read_parquet(OUT / "cities_clusters.parquet")
    clusters = clusters[clusters.date_filed >= "2020-01-01"]
    ops = pd.read_parquet(OUT / "cities_opinions.parquet", columns=["cluster_id", "text"])
    ops = ops[ops.cluster_id.isin(clusters.id)].drop_duplicates("cluster_id").merge(
        clusters[["id", "court_id", "case_name"]], left_on="cluster_id", right_on="id")
    for city, (courts, county, judge) in CITIES.items():
        mine = ops[ops.court_id.str.startswith(tuple(courts))]
        head = mine.text.str.slice(0, 6000)
        local = mine[head.str.contains(county, regex=True)]
        hits = local.text.str.slice(0, 6000).str.extract(judge).bfill(axis=1).iloc[:, 0] if len(local) else pd.Series(dtype=str)
        named = hits.notna().sum()
        print(f"\n{city}: {len(mine):,} decisions since 2020 in {', '.join(courts)}; {len(local):,} from {county.split('(')[0] or county}; "
              f"trial judge named in {named:,} ({named / max(len(local), 1):.0%})")
        for name, case in list(zip(hits.dropna(), local.loc[hits.dropna().index, "case_name"]))[:4]:
            print(f"   {case[:55]:<57} judge: {name.strip()}")
        miss = local[hits.isna()] if len(local) else local
        if len(miss):
            print("   example with no match:", re.sub(r"\s+", " ", miss.text.iloc[0][:300]))


if __name__ == "__main__":
    main()
