"""Build the shareable judges page from the ballot report.

    python -m ballot.build_ballot_report    # first, to refresh the profiles
    python -m ballot.build_page             # writes site/judges.html and site/web/public/index.html

The page is one self-contained HTML file with the profile data embedded. judges.html is the
page body for the Claude artifact (which adds the document wrapper itself); public/index.html
is the same page as a complete document for self-hosting (e.g. the Heroku app in site/web).
"""
from datetime import date
import json

from ballot.build_judge_profiles import OUT, ROOT

REPORT = OUT / "cook_2026_retention_judges.json"
TEMPLATE = ROOT / "site" / "template.html"
PAGE = ROOT / "site" / "judges.html"
STANDALONE = ROOT / "site" / "web" / "public" / "index.html"
_WRAPPER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Sourced profiles of the judges on the Cook County retention ballot.">
<style>html{color-scheme:light dark}body{margin:0}img{max-width:100%}</style>
</head>
<body>
{page}
</body>
</html>
"""

# Fields the page uses; everything else in the report stays out of the published file.
_SENTENCING_DETAILS = ["n_cases", "felony_share", "prison", "treatment", "prison_length_ratio",
                       "prison_by_race", "black_white_prison_gap"]


def page_data(report: dict) -> dict:
    judges = []
    for j in sorted(report["judges"], key=lambda j: j["ballot_number"] or 10_000):
        s = dict(j["sentencing"])
        if "details" in s:
            s["details"] = {k: s["details"][k] for k in _SENTENCING_DETAILS if k in s["details"]}
        judges.append({
            "ballot_number": j["ballot_number"], "name": j["name"], "court": j["court"],
            "assignment": j["assignment"],
            "bar_ratings": [{k: r[k] for k in ("source", "rating", "bars_saying_no") if k in r}
                            for r in j["bar_ratings"]],
            "criminal_court_experience": j["criminal_court_experience"],
            "background": j["background"], "attorney_criticism": j["attorney_criticism"],
            "attorney_feedback": j["attorney_feedback"],
            "isba_evaluation": j["isba_evaluation"], "sentencing": s,
        })
    return {"source": report["source"], "since": report["since"], "data_through": report["data_through"],
            "built": date.today().strftime("%B %-d, %Y"), "judges": judges,
            "population": report["population"]}


def main() -> None:
    data = json.dumps(page_data(json.loads(REPORT.read_text())), separators=(",", ":"))
    data = data.replace("</", "<\\/")  # can't close the <script> tag early
    page = TEMPLATE.read_text().replace("/*DATA*/", data)
    PAGE.write_text(page)
    STANDALONE.parent.mkdir(parents=True, exist_ok=True)
    STANDALONE.write_text(_WRAPPER.replace("{page}", page))
    for path in (PAGE, STANDALONE):
        print(f"{path} ({path.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
