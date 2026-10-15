"""Adapter: Cook County State's Attorney sentencing data -> common sentencing schema.

Source: https://datacatalog.cookcountyil.gov/d/tg8v-tm6u (one row per charge).
Output: one row per defendant per case, using the primary charge's original sentence.
See ballot.judges.metrics for the common schema.
"""
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

JURISDICTION = "Cook County, IL"
SOURCE_URL = "https://datacatalog.cookcountyil.gov/d/tg8v-tm6u"
_CSV_ENDPOINT = "https://datacatalog.cookcountyil.gov/resource/tg8v-tm6u.csv"
# The portal notes the dataset "is no longer actively maintained as of 12/30/2024" (the office
# moved to its own dashboards in 2025). The handful of later sentence dates are typos, e.g. a
# 2026 sentence on a case disposed in 2016.
DATA_END = "2024-12-31"

# Most to least serious. M = first-degree murder, X/1-4 = felonies, A-C = misdemeanors.
CLASS_ORDER = ["M", "X", "1", "2", "3", "4", "A", "B", "C"]

_RACE = {
    "Black": "Black",
    "White": "White",
    "White [Hispanic or Latino]": "Latino",
    "White/Black [Hispanic or Latino]": "Latino",
    "HISPANIC": "Latino",
    "Asian": "Asian",
    "ASIAN": "Asian",
}

# Sentences built around treatment or a second chance rather than punishment.
_TREATMENT = {
    "Mental Health Probation",
    "Drug Court Probation",
    "Veteran's Court Probation",
    "710/410 Probation",  # first-offender drug probation; record can be cleared
    "2nd Chance Probation",
    "Inpatient Mental Health Services",
    "Drug School",
}

_MONTHS_PER_UNIT = {"Year(s)": 12.0, "Months": 1.0, "Weeks": 12 / 52, "Days": 12 / 365}


def fetch(dest: Path, refresh: bool = False) -> Path:
    """Download the full dataset as CSV (~175 MB) unless it's already cached."""
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"$limit": 1_000_000, "$order": "case_participant_id"})
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(f"{_CSV_ENDPOINT}?{query}", tmp)
    tmp.rename(dest)
    return dest


def _canonical_judge_names(names: pd.Series) -> pd.Series:
    """Collapse spacing and middle-initial variants ("William  Raines" / "William B Raines")
    to the most common spelling sharing the same first and last name."""
    clean = names.str.replace(r"\s+", " ", regex=True).str.strip()
    clean = clean.str.replace(r"^([^,]+),\s*(.+)$", r"\2 \1", regex=True)  # "Byrne, Thomas"
    clean = clean.where(clean != clean.str.upper(), clean.str.title())  # "URSULA WALOWSKI"
    parts = clean.str.replace(".", "", regex=False).str.lower().str.split()
    key = parts.str[0] + "|" + parts.str[-1]
    most_common = clean.groupby(key).agg(lambda s: s.value_counts().index[0])
    return key.map(most_common).where(clean.notna())


def load(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = raw[
        (raw.primary_charge == True)  # noqa: E712 - column is bool with NaNs
        & (raw.current_sentence == True)  # noqa: E712
        & (raw.sentence_phase == "Original Sentencing")
    ].copy()

    df["sentence_date"] = pd.to_datetime(df.sentence_date, errors="coerce")
    # The source has typo dates as early as 1930 and as late as 2921.
    df = df[df.sentence_date.between("2000-01-01", DATA_END)]

    df["judge"] = _canonical_judge_names(df.sentence_judge)
    df["offense_class"] = df.disposition_charged_class.astype(str).str.strip().str.upper()
    df = df[df.judge.notna() & df.offense_class.isin(CLASS_ORDER)]

    # "PROMIS Conversion" marks records migrated from a legacy system with no category.
    category = df.updated_offense_category.where(df.updated_offense_category != "PROMIS Conversion")
    df["offense_category"] = category

    df["race"] = df.race.map(_RACE).fillna("Other/Unknown")
    df["prison"] = df.sentence_type == "Prison"
    df["treatment"] = df.commitment_type.isin(_TREATMENT) | df.sentence_type.isin(_TREATMENT)

    term = pd.to_numeric(df.commitment_term, errors="coerce")
    months = term * df.commitment_unit.map(_MONTHS_PER_UNIT)
    df["life_sentence"] = df.prison & (df.commitment_unit == "Natural Life")
    df["prison_months"] = months.where(df.prison & months.between(0, 1200, inclusive="right"), np.nan)

    # Defensive: keep the most serious charge if a defendant still has several rows.
    df["_rank"] = df.offense_class.map({c: i for i, c in enumerate(CLASS_ORDER)})
    df = df.sort_values(["_rank", "prison_months"], ascending=[True, False])
    df = df.drop_duplicates("case_participant_id")

    return df[
        [
            "case_id", "judge", "sentence_date", "offense_class", "offense_category",
            "race", "prison", "treatment", "prison_months", "life_sentence", "court_name",
        ]
    ].reset_index(drop=True)
