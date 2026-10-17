"""Adapter: facts from Injustice Watch's 2026 Cook County judicial election guide.

The guide (2026retention.injusticewatch.org) loads one public JSON file of every race and
candidate. Its written content (notable cases, experience narratives, rating digests, survey
excerpts) is Injustice Watch's copyrighted reporting, so this keeps only facts and labels:
when the judge took the bench, law school, age, the guide's flags ("Former prosecutor",
"Notable reversals"), endorsements, and ratings from groups our other sources don't cover.
Their prose stays on their site, behind the link each profile carries.

The downloaded file lives in data/raw (not committed) and is cited, not redistributed.
"""
import json
import urllib.request
from pathlib import Path

GUIDE = "https://2026retention.injusticewatch.org"
DATA_URL = f"{GUIDE}/cacheddata/current-en.json"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128 Safari/537.36"
# Ratings our bar-ratings adapter already has, directly or through the Alliance.
_COVERED = {"ISBA", "CBA", "AABAR", "HBA", "PRBA", "BWLA", "LAGBAC", "HLAI", "WBAI", "DSL", "BMLA", "AABA", "CCBA"}


def fetch(dest: Path, refresh: bool = False) -> Path:
    if refresh or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": _UA})
        dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return dest


def _slug(name: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s)).strip("-")


def load(path: Path) -> list[dict]:
    """One dict per candidate: facts only, plus the guide URL."""
    out = []
    for race in json.loads(path.read_text()):
        r = race["fields"]
        for c in r["candidates"]:
            f = c["fields"]
            out.append({
                "name": f["nameFormatted"],
                "race_type": "retention" if f["raceType"] == "Retention" else "open seat",
                "race": r["name"], "court": r["level"],
                "url": f"{GUIDE}/candidate/{_slug(f['nameFormatted'])}-{f['candidateId']}",
                "judge_since": f.get("judgeSince"),
                "law_school": f.get("lawSchool"), "graduation_year": f.get("graduationYear"),
                "age": f.get("age"),
                "flags": [x["fields"]["name"] for x in sorted(f.get("flags", []), key=lambda x: x["fields"].get("sortPriority", 99))],
                "flag_slugs": [x["fields"]["slug"] for x in f.get("flags", [])],
                "endorsements": [x["fields"]["name"] for x in f.get("endorsements", [])],
                "other_ratings": [{"source": x["fields"]["ratingAgency"], "rating": x["fields"]["rating"]}
                                  for x in f.get("ratings", []) if x["fields"]["slug"] not in _COVERED],
                "has_notable": bool(f.get("notable")),
                "has_survey": bool(f.get("survey") or f.get("surveyExcerpt")),
            })
    if not out:
        raise ValueError(f"No candidates in {DATA_URL}; the guide's data format may have changed")
    return out


def find(guide: list[dict], name: str, race_type: str) -> dict | None:
    from ballot.judges.match import matches
    hits = [g for g in guide if g["race_type"] == race_type and matches(name, g["name"])]
    return hits[0] if len(hits) == 1 else None
