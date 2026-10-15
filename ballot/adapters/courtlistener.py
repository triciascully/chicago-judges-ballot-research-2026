"""Adapter: Illinois Appellate Court decisions reviewing a Cook County judge, via CourtListener.

Illinois appellate decisions name the trial judge in their header ("Appeal from the Circuit
Court of Cook County ... The Honorable Jane Q. Doe, Judge, presiding."), so each judge's
appealed rulings can be found by full-text search and then confirmed against that header.

Needs a free CourtListener API token in the environment or in the project's .env file:
    COURTLISTENER_TOKEN=...
Anonymous requests work but are limited to 5 a minute.

    python -m ballot.adapters.courtlistener --judge "Lewis Michael Nixon"   # try one judge
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

from ballot.judges.match import name_parts

API = "https://www.courtlistener.com/api/rest/v4"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "courtlistener"


def _token() -> str | None:
    if os.environ.get("COURTLISTENER_TOKEN"):
        return os.environ["COURTLISTENER_TOKEN"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("COURTLISTENER_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def _get(url: str) -> dict:
    """GET with the token if there is one; waits and retries when throttled."""
    headers = {"User-Agent": "ballot-research (voter guide)"}
    if token := _token():
        headers["Authorization"] = f"Token {token}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError("CourtListener needs an API token for opinion text: add "
                                   "COURTLISTENER_TOKEN=... to the project's .env file") from e
            if e.code not in (429, 502, 503):
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Still throttled after retries: {url}")


def name_variants(ballot_name: str) -> list[str]:
    """Ways an opinion header may write the judge's name: with or without a middle initial,
    and with a hyphenated surname or either part of it."""
    clean = re.sub(r"[“”\"']", "", ballot_name).replace(",", "")
    parts = [p for p in clean.split() if p.rstrip(".").lower() not in {"jr", "sr", "ii", "iii"}]
    first, last = parts[0], parts[-1]
    middles = [p for p in parts[1:-1]]
    variants = {f"{first} {last}"}
    for m in middles:
        variants.add(f"{first} {m} {last}")
        variants.add(f"{first} {m[0]}. {last}")
    if "-" in last:
        variants |= {f"{first} {p}" for p in last.split("-")} | {f"{first} {last.replace('-', ' ')}"}
    return sorted(variants)


def search(ballot_name: str, since: str = "2015-01-01") -> list[dict]:
    """Candidate decisions: Cook County appeals whose text contains the judge's name."""
    names = " OR ".join(f'"{v}"' for v in name_variants(ballot_name))
    q = f'"Circuit Court of Cook County" AND ({names}) AND presiding'
    url = f"{API}/search/?" + urllib.parse.urlencode(
        {"q": q, "type": "o", "court": "illappct", "filed_after": since, "order_by": "dateFiled asc"})
    results = []
    while url:
        page = _get(url)
        results += page.get("results", [])
        url = page.get("next")
    return results


def opinion_text(opinion_id: int) -> str:
    """Plain text of one opinion, cached on disk."""
    path = CACHE / "opinions" / f"{opinion_id}.txt"
    if path.exists():
        return path.read_text()
    op = _get(f"{API}/opinions/{opinion_id}/?fields=plain_text,html_with_citations,html")
    text = op.get("plain_text") or re.sub(r"<[^>]+>", " ", op.get("html_with_citations") or op.get("html") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def presided(text: str, ballot_name: str) -> bool:
    """True if the opinion's header names this judge as the trial judge. The header sits in the
    first few thousand characters; looking only there avoids judges merely cited in the body."""
    head = re.sub(r"\s+", " ", text[:6000])
    if not re.search(r"Cook County", head, re.I):
        return False
    given, surnames = name_parts(ballot_name)
    for m in re.finditer(r"Honorable\s+([^,]{3,60}?),?\s+Judges?,?\s+presiding", head, re.I):
        g, s = name_parts(m.group(1))
        if s & surnames and (not g or any(x[:3] == y[:3] for x in g for y in given)):
            return True
    return False


# The disposition is the last word of an Illinois decision ("Affirmed." / "Reversed and
# remanded."), and Rule 23 orders also state it up front in a "Held:" summary.
_OUTCOMES = [
    ("affirmed in part", r"affirmed in part|reversed in part|vacated in part"),
    ("reversed", r"\breversed\b"),
    ("vacated", r"\bvacated\b"),
    ("dismissed", r"appeal dismissed|\bdismissed\b"),
    ("affirmed", r"\baffirmed\b|\baffirm\b"),
]


def outcome(text: str) -> str | None:
    """Disposition from the decision's closing lines, falling back to the "Held:" summary.
    None when neither says, which leaves the case for the model (and for hand labels)."""
    flat = re.sub(r"\s+", " ", text).strip()
    tail = flat[-400:].lower()
    held = re.search(r"held:(.{0,400})", flat[:4000], re.I)
    for chunk in (tail, held.group(1).lower() if held else ""):
        for label, pattern in _OUTCOMES:
            if re.search(pattern, chunk):
                return label
    return None


def judge_decisions(ballot_name: str, since: str = "2015-01-01") -> list[dict]:
    out, seen = [], set()
    for r in search(ballot_name, since):
        key = (r.get("caseName"), r.get("dateFiled"), r.get("docketNumber"))
        if key in seen:  # the same decision can be indexed more than once
            continue
        seen.add(key)
        for op in r.get("opinions", [])[:1]:
            text = opinion_text(op["id"])
            if not presided(text, ballot_name):
                continue
            out.append({
                "judge": ballot_name, "case_name": r.get("caseName"), "date": r.get("dateFiled"),
                "docket": r.get("docketNumber"), "status": r.get("status"),
                "opinion_id": op["id"], "url": "https://www.courtlistener.com" + r.get("absolute_url", ""),
                "outcome": outcome(text), "criminal": bool(re.match(r"People v\.?", r.get("caseName") or "")),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--since", default="2015-01-01")
    args = parser.parse_args()
    print("token:", "found" if _token() else "none (anonymous, 5 requests a minute)")
    print("name variants:", name_variants(args.judge))
    decisions = judge_decisions(args.judge, args.since)
    for d in decisions:
        print(f"  {d['date']}  {d['outcome'] or '?':16} {d['status']:12} {d['case_name'][:70]}")
    print(f"{len(decisions)} decisions confirmed for {args.judge}")


if __name__ == "__main__":
    main()
