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


# Donor groups, checked in order; the first match wins. Individuals are grouped by their reported
# employer and occupation, organizations by their name. Law firms stay one group: telling
# plaintiffs' firms from defense firms needs knowledge about each firm that names alone don't give.
DONOR_GROUPS = [
    ("Prosecutors", r"state'?s attorney|\bccsao\b|\bsao\b|prosecutor"),
    ("Public defenders", r"public defender|appellate defender"),
    ("Police and law enforcement", r"fraternal order of police|\bfop\b|police|sheriff|corrections officer"),
    ("Judges and court staff", r"\bjudge\b|\bcourt\b|judiciary|clerk of the circuit"),
    ("Labor unions", r"\bunion\b|\blocal \d|liuna|laborers|\biuoe\b|\bibew\b|teamsters|afscme|\bseiu\b"
                     r"|painters|sprinkler fitters|elevator constructors|building trades|pipefitters|carpenters"
                     r"|bricklayers|plumbers|journeymen|district council|regional council|\blu \d|ironworkers|operating engineers"),
    ("Party and ward organizations", r"democrat|republican|\bward\b|township|\bparty\b"),
    ("Advocacy PACs", r"\bpac\b|political action|victory fund|\bfund\b"),
    ("Politicians' committees", r"^(friends|citizens|people|committee|comm|neighbors)\b|\bfriends\b|\b(citizens|people|neighbors) (of|for)\b|\bfor (judge|state|chicago|congress|mayor)\b"
                                r"|\bfor [a-z]+$|for alderman|for commissioner"),
    ("Private law firms", r"attorney|lawyer|\blaw\b|legal|counsel|\bllp\b|&\s*associates|litigat|\bfirm\b"
                          # "Surname & Surname" with no business words: in donations to judges, a law firm
                          r"|^(?!.*(builder|management|company|group|construction|mechanical|realty|capital|productions"
                          r"|properties|sons|co\b|inc\b|iron|metal|steel|venture))[a-z'\-]+( [a-z'\-]+)? & [a-z'\-]+( [a-z'\-]+)?$"),
    ("Government (other)", r"state of illinois|cook county|city of chicago|chicago public schools|village|\bcity\b|county"),
]
OTHER_GROUP = "Businesses and other employers"
UNREPORTED = "Employer not reported"


def donor_group(row) -> str:
    """Group for one receipt (individual or organization). Each field is checked on its own so
    patterns anchored to a whole name (like "Surname & Surname") still work."""
    org_name = str(row.LastOnlyName).strip().strip('"') if not str(row.FirstName).strip() else ""
    # In-kind gifts from people carry a first name; group them by employer like other individuals.
    if row.kind == "individual" or not org_name:
        if _org(row.Employer) in _NOT_EMPLOYERS and not org_name:
            return UNREPORTED
        fields = [row.Employer, row.Occupation, org_name]
    else:
        fields = [org_name]
    fields = [str(f).lower().strip() for f in fields if str(f).strip()]
    for group, pattern in DONOR_GROUPS:
        if any(re.search(pattern, f) for f in fields):
            return group
    return OTHER_GROUP


# Only names that look like organizations are listed by name; some people are filed with no
# first name, and individual donors stay unnamed on the page.
_ORG_WORDS = re.compile(r"\b(?:pac|committee|comm|comte|friends|citizens|people|neighbors|union|local|council|inc|llc|ltd"
                        r"|group|company|co|corp|party|org|fund|association|productions|media|pr|management|partners"
                        r"|foundation|trust|democrat\w*|republican|ward|township|for)\b|&", re.I)


def judge_party(lookup_dir: Path, ballot_names) -> dict[str, str]:
    """Party each judge ran under, from their candidate records ("" if none listed)."""
    cand = _read(lookup_dir / "Candidates.txt")
    judges = cand[cand.Office.str.contains(r"Judge|Justice", case=False)]
    out = {}
    for name in ballot_names:
        hits = judges[[_same_person(name, f, l) for f, l in zip(judges.FirstName, judges.LastName)]]
        out[name] = "/".join(sorted(set(hits.PartyAffiliation) - {""}))
    return out


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
    orgs = from_others[(from_others.kind != "individual") & (from_others.FirstName.str.strip() == "")
                       & from_others.LastOnlyName.str.contains(_ORG_WORDS)]
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
        "by_group": {g: round(float(v), 2) for g, v in
                     from_others.assign(group=[donor_group(x) for x in from_others.itertuples()])
                     .groupby("group").amount.sum().sort_values(ascending=False).items()},
    }


# --- Where the judges' organizational donors give elsewhere ------------------------------------

def donor_org_names(receipts: pd.DataFrame) -> set[str]:
    """Normalized names of the organizations in the judges' donor lists: organizations that gave
    directly, and employers reported by individual donors."""
    orgs = receipts[(receipts.FirstName.str.strip() == "") & receipts.LastOnlyName.str.contains(_ORG_WORDS)]
    names = set(orgs.LastOnlyName.map(_org))
    employers = receipts[receipts.kind == "individual"].Employer.map(_org)
    names |= {e for e in employers if e not in _NOT_EMPLOYERS}
    # Skip employer names shaped like a person ("ALLEN N SCHWARTZ"): looking those up would
    # profile an individual's politics, which stays off this project.
    person = re.compile(r"^[A-Z'\-]+ [A-Z] [A-Z'\-]+( JR| SR| III)?$")
    return {n for n in names if len(n) >= 4 and not person.match(n)}


def fetch_org_giving(org_names: set[str], dest: Path, since: str) -> Path:
    """Stream Receipts.txt and keep contributions *made by* these organizations (donor rows with
    no first name whose normalized name matches) to any committee since `since`."""
    req = urllib.request.Request(f"{BASE}/Receipts.txt", headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=600) as resp:
        reader = csv.reader(io.TextIOWrapper(resp, encoding="latin-1", newline=""), delimiter="\t",
                            quoting=csv.QUOTE_NONE)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        with dest.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["org", "CommitteeID", "RcvDate", "Amount", "D2Part"])
            for row in reader:
                if len(row) < len(header) or row[col["FirstName"]].strip():
                    continue
                if row[col["Archived"]].lower() == "true" or row[col["RcvDate"]][:10] < since:
                    continue
                org = _org(row[col["LastOnlyName"]])
                if org in org_names:
                    writer.writerow([org, row[col["CommitteeID"]], row[col["RcvDate"]][:10],
                                     row[col["Amount"]], row[col["D2Part"]]])
    return dest


def committee_parties(lookup_dir: Path) -> dict[str, str]:
    """Party for each committee: the committee's own listing, else its candidate's most recent."""
    cmte, links, cand = (_read(lookup_dir / n) for n in ("Committees.txt", "CmteCandidateLinks.txt", "Candidates.txt"))
    party = {}
    cand_party = dict(zip(cand.ID, cand.PartyAffiliation))
    for cid, cand_id in zip(links.CommitteeID, links.CandidateID):
        if cand_party.get(cand_id):
            party[cid] = cand_party[cand_id]
    for cid, p in zip(cmte.ID, cmte.PartyAffiliation):
        if p.strip():
            party[cid] = p
    return {k: _party_bucket(v) for k, v in party.items()}


def _party_bucket(p: str) -> str:
    p = p.lower()
    if "democrat" in p:
        return "Democratic"
    if "republican" in p:
        return "Republican"
    return "No party listed"


def org_leanings(giving_path: Path, parties: dict[str, str], exclude_committees: set[str]) -> dict[str, dict]:
    """{org: totals by recipient party} for each organization's giving to committees other than
    the judges' own (so the measure is about where else the money goes)."""
    g = pd.read_csv(giving_path, dtype=str)
    g = g[~g.CommitteeID.isin(exclude_committees)]
    g["amount"] = pd.to_numeric(g.Amount, errors="coerce").fillna(0)
    g["party"] = g.CommitteeID.map(parties).fillna("No party listed")
    out = {}
    for org, rows in g.groupby("org"):
        by = rows.groupby("party").amount.sum()
        dem, rep = float(by.get("Democratic", 0)), float(by.get("Republican", 0))
        out[org] = {"total": round(float(rows.amount.sum()), 2), "democratic": round(dem, 2),
                    "republican": round(rep, 2), "no_party": round(float(by.get("No party listed", 0)), 2),
                    "recipients": int(rows.CommitteeID.nunique()),
                    "dem_share": round(dem / (dem + rep), 3) if dem + rep > 0 else None}
    return out
