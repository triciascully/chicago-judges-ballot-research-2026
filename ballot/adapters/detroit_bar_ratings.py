"""Adapter: Detroit Bar Association judicial candidate ratings, 2020-2024.

The association rates candidates in contested Wayne County races every two years. Its site
blocks automated access, so the copies come from the Internet Archive:
  2020  web.archive.org/web/20201031111116/https://www.detroitlawyer.org/judges-2020/
  2022  web.archive.org/web/20220810113652/.../2022/07/PRESS-RELEASE_...-Ratings-4.pdf
  2024  web.archive.org/web/20240808101603/.../2024/07/Detroit-Bar-Association-Press-Release-7-2024.pdf
Michigan trial judges serve six-year terms, so most 2026 incumbents were last rated in 2020.

    python -m ballot.adapters.detroit_bar_ratings   # writes data/ballot/detroit_bar_ratings.csv
"""
import html
from pathlib import Path
import re

import pandas as pd
import pypdf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "detroit"
OUT = ROOT / "data" / "ballot" / "detroit_bar_ratings.csv"
SOURCES = {
    2020: "https://web.archive.org/web/20201031111116/https://www.detroitlawyer.org/judges-2020/",
    2022: "https://web.archive.org/web/20220810113652/https://www.detroitlawyer.org/wp-content/uploads/2022/07/PRESS-RELEASE_Detroit-Bar-Association-2022-Judicial-Candidate-Evaluation-Committee-Ratings-4.pdf",
    2024: "https://web.archive.org/web/20240808101603/https://www.detroitlawyer.org/wp-content/uploads/2024/07/Detroit-Bar-Association-Press-Release-7-2024.pdf",
}
RATINGS = ["Outstanding", "Well Qualified", "Qualified", "Not Qualified", "No Rating"]
_RATING = re.compile(r"(Outstanding|Well Qualified|Not Qualified|Qualified|No Rating[^*]*?)\s*(\*.*)?$")


def _normalize_rating(text: str) -> str:
    return next(r for r in RATINGS if text.startswith(r))


def _race(line: str) -> bool:
    return bool(re.search(r"JUDGE OF|JUSTICE", line)) and line.upper() == line


def parse_2020(path: Path) -> list[dict]:
    """The 2020 ratings are an HTML page: a race heading, then alternating name / rating lines."""
    page = re.sub(r"<script.*?</script>|<style.*?</style>", "", path.read_text(errors="replace"), flags=re.S)
    lines = [l.strip() for l in html.unescape(re.sub(r"<[^>]+>", "\n", page)).split("\n") if l.strip()]
    lines = lines[lines.index("Candidate Rating") + 1:]
    rows, race, name = [], None, None
    for line in lines:
        if line.startswith("*ITALIC"):
            break
        if _race(line):
            race, name = line, None
        elif re.match(r"^\d+ YEAR TERM", line) and race:
            race += " " + line
        elif name is None:
            name = line
        else:
            rows.append({"year": 2020, "race": race, "name": name.rstrip("*"),
                         "incumbent_then": name.endswith("*"), "rating": _normalize_rating(line), "note": ""})
            name = None
    return rows


def parse_release(path: Path, year: int) -> list[dict]:
    """The 2022 and 2024 ratings are PDF press releases: a race heading, then "Name  Rating" lines."""
    text = "\n".join(p.extract_text(extraction_mode="layout") for p in pypdf.PdfReader(path).pages)
    rows, race = [], None
    for raw in text.split("\n"):
        line = re.sub(r"\s{2,}", "  ", raw).strip()
        if _race(line):
            race = line
            continue
        if not race or "  " not in line or line.startswith("Remaining"):
            continue
        name, rest = line.split("  ", 1)
        m = _RATING.match(rest.strip())
        if not m:
            continue
        rows.append({"year": year, "race": race, "name": re.sub(r"\s*\(I\)$", "", name).strip(),
                     "incumbent_then": name.strip().endswith("(I)"), "rating": _normalize_rating(m.group(1)),
                     "note": (m.group(2) or "").strip("* ").replace("  ", " ")})
    return rows


def load() -> pd.DataFrame:
    rows = parse_2020(RAW / "dba_2020.html")
    for year in (2022, 2024):
        rows += parse_release(RAW / f"dba_{year}.pdf", year)
    df = pd.DataFrame(rows)
    df["source"] = df.year.map(SOURCES)
    return df


def main() -> None:
    df = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.groupby(["year", "rating"]).size().unstack(fill_value=0).to_string())
    print(f"\n{len(df)} ratings -> {OUT}")


if __name__ == "__main__":
    main()
