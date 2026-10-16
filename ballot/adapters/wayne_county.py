"""Adapter: Wayne County, Michigan (Detroit) judicial races.

Source: the Wayne County Clerk's candidate list for the Nov. 3, 2026 general election
(PDF, data/raw/detroit/candidates_2026.pdf). Michigan judges run in nonpartisan contested
elections; incumbents are marked on the ballot.

    python -m ballot.adapters.wayne_county      # writes data/ballot/wayne_2026_judicial.csv
"""
from pathlib import Path
import re

import pandas as pd
import pypdf

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "raw" / "detroit" / "candidates_2026.pdf"
OUT = ROOT / "data" / "ballot" / "wayne_2026_judicial.csv"
SOURCE = ("https://www.waynecountymi.gov/files/assets/mainsite/v/1/clerk/documents/elections/"
          "unofficial-candidate-list-1103202698.pdf")

_RACE = re.compile(r"^(?P<race>.*?(?:JUSTICE|JUDGE).*?)\s*\((?P<seats>\d+)\)\s*POSITIONS?$")
_STATUS = re.compile(r"^(Incum\.|Incumbent|Non-Incumbent)\s+")


def _court(race: str) -> str:
    r = race.upper()
    if "SUPREME" in r:
        return "Michigan Supreme Court"
    if "APPEALS" in r:
        return "Michigan Court of Appeals, 1st District"
    if "CIRCUIT" in r:
        return "3rd Circuit Court (Wayne County)"
    if "PROBATE" in r:
        return "Wayne County Probate Court"
    m = re.search(r"(\d+[A-Z]?)(?:ST|ND|RD|TH)?\s+DISTRICT JUDGE OF DISTRICT COURT", r)
    return f"{m.group(1)} District Court" if m else race.title()


def parse(pdf: Path = PDF) -> pd.DataFrame:
    """One row per candidate in a judicial race, from the clerk's layout-preserved text."""
    rows, race = [], None
    for page in pypdf.PdfReader(pdf).pages:
        for raw in page.extract_text(extraction_mode="layout").split("\n"):
            line = re.sub(r"\s{2,}", "  ", raw).strip()
            if not line:
                continue
            if m := _RACE.match(line):
                race = {"race": re.sub(r"\s+", " ", m["race"]).strip(" -"), "seats": int(m["seats"])}
                continue
            if re.match(r"^(PARTY\b|DATE$|WITHDRAWAL)", line):  # column labels at the top of each page
                continue
            if race is None or re.search(r"POSITIONS?$|OFFICE OF|GENERAL ELECTION|CANDIDATE LIST|Page \d+", line):
                if re.search(r"POSITIONS?$", line):
                    race = None  # a non-judicial race heading ends the judicial block
                continue
            status = _STATUS.match(line)
            fields = line[status.end():].split("  ") if status else line.split("  ")
            name = fields[0].strip()
            if not re.match(r"^[A-Z][A-Za-z.'\- ,]+$", name) or name.lower().startswith("no candidate"):
                continue
            rows.append({
                "court": _court(race["race"]), "race": race["race"], "seats": race["seats"],
                "name": name.rstrip(","), "incumbent": bool(status and status.group(1) != "Non-Incumbent"),
                "disqualified": "DISQ" in line.split("  ")[-1:],
                "detroit": not re.search(r"\d+[A-Z]? District Court$", _court(race["race"]))
                           or _court(race["race"]).startswith("36 "),
                "source": SOURCE,
            })
    df = pd.DataFrame(rows)
    df = df[~df.disqualified].drop(columns="disqualified").reset_index(drop=True)
    df["contested"] = df.groupby("race").name.transform("count") > df.seats
    return df


def main() -> None:
    df = parse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.groupby(["court", "race"]).agg(seats=("seats", "first"), candidates=("name", "size"),
                                            incumbents=("incumbent", "sum")).to_string())
    print(f"\n{len(df)} candidates ({df.detroit.sum()} on Detroit ballots) -> {OUT}")


if __name__ == "__main__":
    main()
