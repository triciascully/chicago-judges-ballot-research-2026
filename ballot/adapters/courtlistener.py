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
_MIN_INTERVAL = 4.0  # seconds between requests
_last_request = 0.0


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
    """GET with the token if there is one. When throttled, waits as long as the server's
    Retry-After says (search is limited to 50 requests an hour) and tries again."""
    headers = {"User-Agent": "ballot-research (voter guide)"}
    if token := _token():
        headers["Authorization"] = f"Token {token}"
    global _last_request
    for attempt in range(8):
        # Space requests out: the API also has a short-term limit of a few requests per few seconds.
        time.sleep(max(0.0, _MIN_INTERVAL - (time.monotonic() - _last_request)))
        _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError("CourtListener needs an API token for opinion text: add "
                                   "COURTLISTENER_TOKEN=... to the project's .env file") from e
            if e.code not in (429, 502, 503):
                raise
            wait = int(e.headers.get("Retry-After") or 30 * (attempt + 1))
            print(f"  throttled; waiting {wait // 60} min {wait % 60} s", flush=True)
            time.sleep(wait + 5)
        except (TimeoutError, ConnectionError, urllib.error.URLError) as e:
            wait = 30 * (attempt + 1)
            print(f"  network error ({type(e).__name__}); retrying in {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Gave up after repeated throttling: {url}")


def _cached_get(url: str) -> dict:
    """_get, cached on disk by URL, so reruns never spend the hourly search budget twice."""
    import hashlib
    path = CACHE / "search" / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")
    if path.exists():
        return json.loads(path.read_text())
    data = _get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


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


def search(ballot_names: list[str], since: str = "2015-01-01") -> list[dict]:
    """Candidate decisions for several judges in one query: Cook County appeals whose text
    contains any of their names. Batching matters because search allows 50 requests an hour.
    Asks for unpublished (Rule 23) orders too, which are most Illinois appellate decisions."""
    names = " OR ".join(f'"{v}"' for n in ballot_names for v in name_variants(n))
    q = f'"Circuit Court of Cook County" AND ({names}) AND presiding'
    url = f"{API}/search/?" + urllib.parse.urlencode(
        {"q": q, "type": "o", "court": "illappct", "filed_after": since, "order_by": "dateFiled asc",
         "stat_Published": "on", "stat_Unpublished": "on", "stat_Errata": "on", "stat_Separate": "on",
         "stat_In-chambers": "on", "stat_Relating-to": "on", "stat_Unknown": "on"})
    results = []
    while url:
        page = _cached_get(url)
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
    """True if the opinion's header names this judge as the trial judge.

    Headers print the caption in two columns, and text extraction interleaves them
    ("The Honorable ATATRUK PROPERTIES, INC. ) Caroline Kate Moreland, ) Judge, presiding."),
    so instead of matching one phrase this looks for the judge's surname and a given name in
    the 300 characters before "Judge, presiding". Only the first few thousand characters are
    checked, so judges merely cited in the body don't count."""
    head = re.sub(r"\s+", " ", text[:8000])
    if not re.search(r"Cook County", head, re.I):
        return False
    given, surnames = name_parts(ballot_name)
    for m in re.finditer(r"Judges?,?\s+presiding", head, re.I):
        window = head[max(0, m.start() - 300):m.start()].lower()
        if not re.search(r"\bhon(orable|\.)", window):
            continue
        if any(re.search(rf"\b{re.escape(s)}\b", window) for s in surnames) and \
                any(re.search(rf"\b{re.escape(g[:3])}", window) for g in given):
            return True
    return False


# Where an Illinois decision states its disposition, most reliable first:
#   1. the last numbered paragraph that is just the disposition ("¶ 68 Affirmed.",
#      "¶ 31 Reversed and remanded."); newer opinions add an attorneys page after it
#   2. the "Judgment: Affirmed." line in a published opinion's summary page
#   3. a Rule 23 order's "Held:" summary
#   4. the court's own sentence ("we affirm the judgment of the circuit court")
_OUTCOMES = [
    ("affirmed in part", r"affirmed in part|reversed in part|vacated in part|affirm in part|reverse in part"),
    ("reversed", r"\breverse"),
    ("vacated", r"\bvacate"),
    ("dismissed", r"dismiss"),
    ("affirmed", r"\baffirm"),
]
_DISPOSITION = r"(?:affirmed|reversed|vacated|dismissed|appeal dismissed|remanded|judgment affirmed)[^.¶]{0,120}"


def _label(snippet: str) -> str | None:
    s = snippet.lower()
    return next((label for label, pattern in _OUTCOMES if re.search(pattern, s)), None)


def outcome(text: str) -> str | None:
    """Disposition of the appeal, or None when the text doesn't state one clearly (those cases
    are left for the model and for hand labels)."""
    flat = re.sub(r"\s+", " ", text)
    paragraphs = list(re.finditer(rf"¶\s*\d+\s+({_DISPOSITION})\.", flat, re.I))
    if paragraphs:
        return _label(paragraphs[-1].group(1))
    if m := re.search(rf"Judgment\s*:?\s*({_DISPOSITION})", flat[:6000], re.I):
        return _label(m.group(1))
    if m := re.search(r"Held:(.{0,400})", flat[:5000], re.I):
        if label := _label(m.group(1)):
            return label
    sentences = re.findall(r"[^.]*\bwe (?:affirm|reverse|vacate|dismiss)[^.]*\.", flat[-4000:], re.I)
    return _label(sentences[-1]) if sentences else None


def decisions_for(ballot_names: list[str], since: str = "2015-01-01") -> list[dict]:
    """Confirmed decisions for a batch of judges; each opinion goes to the judge its header
    names as presiding (opinion downloads don't count against the search limit)."""
    out, seen = [], set()
    for r in search(ballot_names, since):
        key = (r.get("caseName"), r.get("dateFiled"), r.get("docketNumber"))
        if key in seen:  # the same decision can be indexed more than once
            continue
        seen.add(key)
        for op in r.get("opinions", [])[:1]:
            text = opinion_text(op["id"])
            for name in ballot_names:
                if presided(text, name):
                    out.append({
                        "judge": name, "case_name": r.get("caseName"), "date": r.get("dateFiled"),
                        "docket": r.get("docketNumber"), "status": r.get("status"),
                        "opinion_id": op["id"], "url": "https://www.courtlistener.com" + r.get("absolute_url", ""),
                        "outcome": outcome(text), "criminal": bool(re.match(r"People v\.?", r.get("caseName") or "")),
                    })
                    break
    return out


def judge_decisions(ballot_name: str, since: str = "2015-01-01") -> list[dict]:
    return decisions_for([ballot_name], since)


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
