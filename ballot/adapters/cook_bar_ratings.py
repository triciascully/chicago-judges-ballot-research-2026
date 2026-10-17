"""Adapter: Cook County bar association ratings for judicial retention candidates.

Sources (one row per source per judge in the output):
  - Alliance of Bar Associations (voteforjudges.org): which of ~13 bars say no
  - Illinois State Bar Association: Yes/No plus a written evaluation
  - Chicago Bar Association: Qualified / Not Recommended, plus ballot number

These are human-written pages, so the parsers are tied to their current layout
and fail loudly (rather than return partial data) if it changes.
"""
import html
import re
from pathlib import Path
import urllib.request

import pandas as pd

ALLIANCE_URL = "https://www.voteforjudges.org/"
ISBA_URL = "https://www.isba.org/judicial-elections/counties/cook"
CBA_URL = "https://www.chicagobar.org/CBA/CBA/JEC/Judicial_Voters_Guide.aspx"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128 Safari/537.36"


def _fetch(url: str, dest: Path, refresh: bool) -> str:
    if refresh or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return dest.read_text(errors="replace")


def _text_lines(page: str) -> list[str]:
    page = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", page))
    return [line.strip() for line in text.split("\n") if line.strip()]


def parse_alliance(page: str) -> pd.DataFrame:
    lines = _text_lines(page)
    start, end = lines.index("Alliance of Bar Associations for Judicial Screening"), lines.index("Ratings Key")
    bars = {lines[i].rstrip(":"): lines[i + 1] for i in range(start, end) if re.fullmatch(r"[A-Z]+:", lines[i])}

    rows = []
    for i, line in enumerate(lines):
        if line == "All reporting bars say yes.":
            no = []
        elif m := re.fullmatch(r"(.+) says? no\. All other reporting bars say yes\.", line):
            no = re.split(r",\s*(?:and\s+)?|\s+and\s+", m[1])
        else:
            continue
        unknown = set(no) - bars.keys()
        if unknown:
            raise ValueError(f"Unrecognized bar abbreviations {unknown} on {ALLIANCE_URL}")
        rows.append({
            "source": "Alliance of Bar Associations", "name": lines[i - 1],
            "rating": "all yes" if not no else f"{len(no)} of {len(bars)} say no",
            "bars_saying_no": "; ".join(f"{b} ({bars[b]})" for b in no), "summary": line, "url": ALLIANCE_URL,
        })
    return pd.DataFrame(rows)


def parse_isba(page: str) -> pd.DataFrame:
    lines = _text_lines(page)
    lines = lines[lines.index("Retention"):]  # contested races come first; skip them
    rating = re.compile(r"[–-]\s*(Yes|No)")
    heads = [i for i in range(1, len(lines)) if rating.fullmatch(lines[i])]
    rows = []
    for j, i in enumerate(heads):
        # The evaluation runs until the next judge's name line, or the "ISBA recommends" sentence.
        stop = heads[j + 1] - 1 if j + 1 < len(heads) else len(lines)
        body = " ".join(lines[i + 1:stop])
        body = re.split(r"(?<=\.)\s*ISBA\s+(?:recommends|does not recommend)", body)[0].strip()
        rows.append({"source": "Illinois State Bar Association", "name": lines[i - 1],
                     "rating": rating.fullmatch(lines[i])[1], "summary": body, "url": ISBA_URL})
    return pd.DataFrame(rows)


def parse_cba(page: str) -> pd.DataFrame:
    lines = _text_lines(page)
    start = next(i for i, l in enumerate(lines) if l.endswith("RETENTION") and "COURT" in l)
    text = " ".join(lines[start:])
    rows = [
        {"source": "Chicago Bar Association", "name": name.replace("''", '"').strip(),
         "rating": {"Q": "Qualified", "NR": "Not Recommended"}[code], "ballot_number": int(num), "url": CBA_URL}
        for num, name, code in re.findall(r"\b(\d{3})\s*\|\s*([^|]+?)\s*\|\s*(Q|NR)\b", text)
    ]
    return pd.DataFrame(rows)


def load(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for url, cache, parse in [(ALLIANCE_URL, "alliance.html", parse_alliance),
                              (ISBA_URL, "isba.html", parse_isba),
                              (CBA_URL, "cba.html", parse_cba)]:
        frame = parse(_fetch(url, cache_dir / cache, refresh))
        if frame.empty:
            raise ValueError(f"No ratings parsed from {url}; the page layout may have changed")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# Contested (open-seat) races. Same three pages; these sections come before the retention ones.

_CODES = {"HQ": "Highly Qualified", "HR": "Highly Recommended", "WQ": "Well Qualified", "R": "Recommended",
          "Q": "Qualified", "NR": "Not Recommended", "NQ": "Not Qualified"}
_NEGATIVE = {"NR", "NQ"}
_SEAT_LINE = re.compile(r"^(?:Vacancy of|Hon\.|Cook Circuit Court|Illinois Appellate Court|Circuit Court\b)|Vacancy$")


def _clean_name(name: str) -> str:
    name = re.sub(r"\s*\((?:D|R|G|L|I)\)$", "", name.strip())
    return re.sub(r"[’‘]\s*", "'", name).replace("“", '"').replace("”", '"').replace("''", '"')


def parse_cba_contested(page: str) -> pd.DataFrame:
    """Ballot order, seat and CBA finding for each open-seat candidate."""
    lines = _text_lines(page)
    start = next(i for i, l in enumerate(lines) if l.startswith("ILLINOIS APPELLATE COURT"))
    stop = next(i for i, l in enumerate(lines) if l.endswith("RETENTION") and "COURT" in l)
    rows, seat = [], None
    for line in lines[start:stop]:
        if m := re.fullmatch(r"#?(\d{3})\s*\|\s*(.+?)\s*\|\s*(HQ|Q|NR)\b.*", line.replace("\xa0", " ")):
            party = re.search(r"\((\w)\)\s*$", m[2])
            rows.append({"source": "Chicago Bar Association", "name": _clean_name(m[2]), "seat": seat,
                         "party": {"D": "Democratic", "R": "Republican"}.get(party[1], party[1]) if party else "",
                         "ballot_number": int(m[1]), "rating": _CODES.get(m[3], m[3]), "url": CBA_URL})
        elif "(Vacancy of" in line or "Converted" in line:
            seat = line.strip()
    return pd.DataFrame(rows)


def parse_isba_contested(page: str) -> pd.DataFrame:
    lines = _text_lines(page)
    lines = lines[:lines.index("Retention")]
    rating = re.compile(r"[–-]\s*(Highly Qualified|Qualified|Not Qualified|Not Recommended)")
    heads = [i for i in range(1, len(lines)) if rating.fullmatch(lines[i])]
    rows = []
    for j, i in enumerate(heads):
        stop = heads[j + 1] - 1 if j + 1 < len(heads) else len(lines)
        body = [l for l in lines[i + 1:stop] if not _SEAT_LINE.search(l) and len(l) > 2]
        rows.append({"source": "Illinois State Bar Association", "name": _clean_name(lines[i - 1]),
                     "rating": rating.fullmatch(lines[i])[1], "summary": " ".join(body), "url": ISBA_URL})
    return pd.DataFrame(rows)


def parse_alliance_contested(page: str) -> pd.DataFrame:
    """The Alliance grid: each candidate's name, then one rating code per member bar."""
    lines = _text_lines(page)
    lines = lines[lines.index("Ratings Key"):]
    code = re.compile(r"(HQ|HR|WQ|R|Q|NR|NQ)?¹?")  # a lone ¹ cell: not evaluated, through no fault of the candidate
    rows = []
    for i, line in enumerate(lines):
        if len(line) < 4 or _SEAT_LINE.search(line) or i + 1 >= len(lines) or not code.fullmatch(lines[i + 1]):
            continue
        if "say yes" in line or "says no" in line or "say no" in line:
            break  # the retention section
        codes = []
        for nxt in lines[i + 1:]:
            if not (m := code.fullmatch(nxt)):
                break
            codes.append(m[1] or "not evaluated")
        if len(codes) < 5:  # a column header of single letters, not a rating row
            continue
        rated = [c for c in codes if c != "not evaluated"]
        neg = [c for c in rated if c in _NEGATIVE]
        rating = ("all rate qualified or better" if not neg
                  # The Alliance rates candidates who skip its process NR, so a solid NR row usually means that.
                  else f"all {len(rated)} say not recommended (the Alliance's rating for candidates who don't take part)"
                  if all(c == "NR" for c in rated)
                  else f"{len(neg)} of {len(rated)} say not qualified or not recommended")
        rows.append({"source": "Alliance of Bar Associations", "name": _clean_name(line), "rating": rating,
                     "summary": ", ".join(codes), "url": ALLIANCE_URL})
    return pd.DataFrame(rows)


def load_contested(cache_dir: Path) -> pd.DataFrame:
    pages = {name: (cache_dir / f"{name}.html").read_text(errors="replace") for name in ("alliance", "isba", "cba")}
    frames = [parse_cba_contested(pages["cba"]), parse_isba_contested(pages["isba"]), parse_alliance_contested(pages["alliance"])]
    for f, src in zip(frames, ("CBA", "ISBA", "Alliance")):
        if f.empty:
            raise ValueError(f"No contested-race ratings parsed from the {src} page; the layout may have changed")
    return pd.concat(frames, ignore_index=True)
