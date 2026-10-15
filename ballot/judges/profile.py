"""Assemble a voter-facing profile for one judge from every source.

All text here comes from fixed templates filled with computed numbers or quotes copied
verbatim from the sources; no language model writes profile text. Each claim carries
what it rests on (a quote, or a 95% range and sample size) so readers can check it.
"""
from ballot.judges import metrics
from ballot.judges.extract import flag_evidence

# Plain-language labels for the background flags, in display order.
BACKGROUND = {
    "former_prosecutor": "Former prosecutor",
    "former_criminal_defense": "Former criminal defense attorney",
    "represented_vulnerable_clients": "Represented children, poor clients, or people with disabilities",
    "treatment_court": "Has presided over a treatment court (mental health, drug, or veterans)",
}


def _pts(x: float) -> str:
    return f"{100 * x:+.1f} points"


def sentencing_findings(p: dict) -> list[str]:
    """Sentences for each measure whose 95% range excludes "no difference"."""
    out = []
    pr = p["prison"]
    if pr["lo"] > 0 or pr["hi"] < 0:
        more = "more" if pr["lo"] > 0 else "less"
        out.append(f"Sent people to prison {more} often than expected for comparable charges: "
                   f"{pr['rate']:.0%} of cases vs {pr['expected']:.0%} expected "
                   f"({_pts(pr['diff'])}; 95% range {_pts(pr['lo'])} to {_pts(pr['hi'])}).")
    tr = p["treatment"]
    if tr["lo"] > 0 or tr["hi"] < 0:
        more = "more" if tr["lo"] > 0 else "less"
        out.append(f"Gave treatment-based sentences (mental health, drug, or veterans court probation, "
                   f"second-chance probation) {more} often than expected: {tr['rate']:.1%} vs "
                   f"{tr['expected']:.1%} ({_pts(tr['diff'])}; 95% range {_pts(tr['lo'])} to {_pts(tr['hi'])}).")
    ln = p.get("prison_length_ratio")
    if ln and (ln["lo"] > 1 or ln["hi"] < 1):
        longer = "longer" if ln["lo"] > 1 else "shorter"
        out.append(f"Prison terms were {longer} than for comparable charges: {ln['ratio']:.2f}x "
                   f"(95% range {ln['lo']:.2f}x to {ln['hi']:.2f}x, {ln['n']:,} prison sentences).")
    gap = p.get("black_white_prison_gap")
    if gap and (gap["lo"] > 0 or gap["hi"] < 0):
        group, other = ("Black", "white") if gap["lo"] > 0 else ("white", "Black")
        out.append(f"{group} defendants went to prison more often than {other} defendants facing comparable "
                   f"charges: {_pts(abs(gap['gap']))} (95% range {_pts(gap['lo'])} to {_pts(gap['hi'])}).")
    return out


def sentencing_section(n_sentences: int, profile: dict | None, since: str, data_through: str) -> dict:
    if profile is None:
        return {"status": "no record", "n_sentences": n_sentences,
                "summary": f"No usable sentencing record in county data ({since[:4]}–{data_through[:4]})."}
    notes = []
    if profile["n_cases"] < metrics.MIN_CASES:
        notes.append(f"Limited record ({profile['n_cases']} sentences); treat these numbers as rough.")
    if profile["likely_specialty_court"]:
        notes.append("Mostly treatment sentences, so likely a specialty court; comparisons with "
                     "general courtrooms are limited.")
    findings = sentencing_findings(profile)
    return {
        "status": "profile" if profile["n_cases"] >= metrics.MIN_CASES else "limited",
        "n_sentences": profile["n_cases"],
        "period": f"{profile['first_sentence']} to {profile['last_sentence']}",
        "findings": findings or ["No sentencing measure stood out from comparable cases."],
        "notes": notes,
        "details": profile,
    }


def background_section(isba_text: str | None, extraction: dict | None, n_sentences: int) -> dict:
    """Career background and attorney criticism from the ISBA evaluation, plus criminal-court
    experience combined with court records."""
    evidence = flag_evidence(extraction, isba_text) if extraction and isba_text else {}
    items = [{"label": label, "evidence": evidence[flag]} for flag, label in BACKGROUND.items()
             if evidence.get(flag)]

    basis = []
    if evidence.get("criminal_bench"):
        basis.append({"source": "ISBA evaluation", "evidence": evidence["criminal_bench"]})
    if n_sentences >= metrics.MIN_GROUP:
        basis.append({"source": "Cook County sentencing records",
                      "evidence": [f"{n_sentences:,} sentences since 2019"]})
    return {
        "criminal_court_experience": {"value": bool(basis), "basis": basis},
        "background": items,
        "attorney_criticism": evidence.get("has_criticism", []),
        "isba_evaluation": isba_text,
    }


def population_point(p: dict, ballot_number: int | None = None) -> dict:
    """One judge's sentencing measures for the comparison chart, in percentage points
    (prison-term length as percent longer or shorter), each as [estimate, low, high]."""
    def pts(m: dict, key: str = "diff") -> list[float]:
        return [round(100 * m[k], 1) for k in (key, "lo", "hi")]

    ln, gap = p.get("prison_length_ratio"), p.get("black_white_prison_gap")
    return {
        "name": p["judge"], "ballot_number": ballot_number, "n": p["n_cases"],
        "last": p["last_sentence"][:7], "specialty": p["likely_specialty_court"],
        "limited": p["n_cases"] < metrics.MIN_CASES,
        "prison": pts(p["prison"]), "treatment": pts(p["treatment"]),
        "length": [round(100 * (ln[k] - 1), 1) for k in ("ratio", "lo", "hi")] if ln else None,
        "gap": pts(gap, "gap") if gap else None,
    }
