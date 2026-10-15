"""Adapter: Illinois State Board of Elections campaign disclosure data -> judges' receipts.

Source: https://elections.il.gov/campaigndisclosuredatafiles/ (tab-separated bulk files,
refreshed daily; field definitions in CampaignDisclosureDataDictionary.txt).

Receipts.txt is about 1 GB, so it is streamed and only rows for the judges' own committees
are kept. Joint committees that raise money for many judges at once are excluded from
per-judge totals, because their money can't be attributed to one judge.
"""
import csv
import io
from pathlib import Path
import re
import urllib.request

import pandas as pd

from ballot.judges.match import name_parts

BASE = "https://elections.il.gov/campaigndisclosuredatafiles"
LOOKUPS = ["Committees.txt", "Candidates.txt", "CmteCandidateLinks.txt"]
JOINT_COMMITTEES = {"6505"}  # Committee for Retention of Judges in Cook County
RECEIPT_FIELDS = ["ID", "CommitteeID", "LastOnlyName", "FirstName", "RcvDate", "Amount", "Occupation",
                  "Employer", "City", "State", "D2Part", "Description", "Archived"]
# The server refuses Python's default user agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (ballot-research; voter guide)"}
D2_PARTS = {"1": "individual", "2": "transfer in", "3": "loan", "4": "other", "5": "in-kind"}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, encoding="latin-1",
                       on_bad_lines="skip").fillna("")


def fetch_lookups(dest: Path, refresh: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in LOOKUPS:
        if refresh or not (dest / name).exists():
            req = urllib.request.Request(f"{BASE}/{name}", headers=_HEADERS)
            (dest / name).write_bytes(urllib.request.urlopen(req, timeout=120).read())


def _same_person(ballot_name: str, first: str, last: str) -> bool:
    """Exact surname match (no typo tolerance: "Malone" is not "Maloney") plus a shared
    given name, tolerating nicknames that are prefixes ("Chris" / "Christopher")."""
    b_given, b_sur = name_parts(ballot_name)
    c_given, c_sur = name_parts(f"{first} {last}".replace('"', ""))
    if not b_sur & c_sur:
        return False
    return any(x.startswith(y) or y.startswith(x) for x in b_given for y in c_given if min(len(x), len(y)) >= 3)


def judge_committees(lookup_dir: Path, ballot_names) -> pd.DataFrame:
    """One row per (ballot judge, committee) for judicial candidates, joint committees excluded."""
    cand, links, cmte = (_read(lookup_dir / n) for n in ("Candidates.txt", "CmteCandidateLinks.txt", "Committees.txt"))
    judges = cand[cand.Office.str.contains(r"Judge|Justice", case=False)]
    rows = []
    for name in ballot_names:
        ids = judges[[_same_person(name, f, l) for f, l in zip(judges.FirstName, judges.LastName)]].ID
        for cid in links[links.CandidateID.isin(ids)].CommitteeID.unique():
            if cid in JOINT_COMMITTEES:
                continue
            c = cmte[cmte.ID == cid].iloc[0]
            rows.append({"judge": name, "committee_id": cid, "committee": c.Name, "status": c.Status})
    return pd.DataFrame(rows)


def fetch_receipts(committee_ids: set[str], dest: Path, since: str) -> Path:
    """Stream Receipts.txt and keep current (non-archived) rows for `committee_ids` since `since`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/Receipts.txt", headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=600) as resp:
        lines = io.TextIOWrapper(resp, encoding="latin-1", newline="")
        reader = csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        with dest.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(RECEIPT_FIELDS)
            for row in reader:
                if len(row) < len(header) or row[col["CommitteeID"]] not in committee_ids:
                    continue
                if row[col["Archived"]].lower() == "true" or row[col["RcvDate"]][:10] < since:
                    continue
                writer.writerow([row[col[f]] for f in RECEIPT_FIELDS])
    return dest


# Donors whose occupation or employer looks like legal work. These are the contributions that
# matter for independence questions, since lawyers can appear before the judge they fund.
_LEGAL = re.compile(r"attorney|lawyer|\blaw\b|legal|counsel|llp|l\.l\.p|\bp\.?c\.?$|litigat|paralegal|judge|court", re.I)


def load_receipts(path: Path, committees: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["amount"] = pd.to_numeric(df.Amount, errors="coerce").fillna(0.0)
    df["date"] = pd.to_datetime(df.RcvDate.str[:10], errors="coerce")
    df["kind"] = df.D2Part.str[:1].map(D2_PARTS).fillna("other")  # codes look like "1A"
    df["legal"] = df.Occupation.str.contains(_LEGAL) | df.Employer.str.contains(_LEGAL)
    return df.merge(committees[["committee_id", "judge"]], left_on="CommitteeID", right_on="committee_id")


_NOT_EMPLOYERS = {"", "RETIRED", "NONE", "N/A", "NA", "SELF", "SELF EMPLOYED", "SELF-EMPLOYED", "HOMEMAKER",
                  "NOT EMPLOYED", "UNEMPLOYED", "INFORMATION REQUESTED", "STUDENT"}


def _org(employer: str) -> str:
    """Normalize an employer name so variants of one firm group together."""
    e = re.sub(r"[.,]", "", employer.upper())
    e = re.sub(r"\b(LLP|LLC|PC|P C|LTD|INC|PLLC|THE|AND|&)\b", " ", e)
    return re.sub(r"\s+", " ", e).strip()


def judge_summary(receipts: pd.DataFrame, judge: str) -> dict:
    """Totals for one judge's own committees. Individual donors' names are left out; employers
    stand in for them, since the question is which firms and organizations fund a judge.

    Illinois reports a donor's employer and occupation mainly for larger contributions, so
    "from_legal" counts only donors who listed a legal job; "with_employer" says how much of
    the money that covers."""
    r = receipts[receipts.judge == judge]
    given, surnames = name_parts(judge)
    is_self = (r.LastOnlyName.str.upper().isin({s.upper() for s in surnames})
               & r.FirstName.str.lower().str[:3].isin({g[:3] for g in given}))
    gifts = r[r.kind != "loan"]
    from_others = gifts[~is_self.loc[gifts.index]]
    self_funded = r[(r.kind == "loan") | is_self]
    individual = from_others[from_others.kind == "individual"]
    with_employer = individual[~individual.Employer.map(_org).isin(_NOT_EMPLOYERS)]
    employers = (with_employer.assign(org=lambda d: d.Employer.map(_org))
                 .groupby("org").agg(total=("amount", "sum"), donors=("amount", "size"), legal=("legal", "max"))
                 .sort_values("total", ascending=False))
    # Organizations only: in-kind gifts from individuals carry a person's name, which stays out.
    orgs = from_others[(from_others.kind != "individual") & (from_others.FirstName.str.strip() == "")]
    organizations = (orgs.assign(org=lambda d: d.LastOnlyName.map(_org))
                     .groupby("org").amount.sum().sort_values(ascending=False))
    money = lambda d: round(float(d.amount.sum()), 2)
    return {
        "raised_from_others": money(from_others),
        "self_funded": money(self_funded),
        "from_individuals": money(individual),
        "with_employer": money(with_employer),
        "from_legal": money(with_employer[with_employer.legal]),
        "from_organizations": money(from_others[from_others.kind != "individual"]),
        "n_contributions": int(len(from_others)),
        "first": r.date.min().date().isoformat() if len(r) else None,
        "last": r.date.max().date().isoformat() if len(r) else None,
        "top_employers": [{"employer": k, "total": round(float(v.total), 2), "donors": int(v.donors),
                           "legal": bool(v.legal)} for k, v in employers.head(8).iterrows()],
        "top_organizations": [{"name": k, "total": round(float(v), 2)} for k, v in organizations.head(5).items()],
    }
