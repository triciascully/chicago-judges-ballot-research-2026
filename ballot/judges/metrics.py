"""Judge sentencing metrics, independent of jurisdiction.

Input is the common sentencing schema that every adapter produces, one row per
defendant per case:
    judge, sentence_date, offense_class, offense_category (may be NaN), race,
    prison (bool), treatment (bool), prison_months (NaN unless a prison term)

Each judge is compared against what the whole court would be expected to do
with the same mix of cases (indirect standardization). Cases are grouped by
offense class x offense category, falling back to class alone when a group is
too small. A judge whose cases are mostly murders isn't "harsh" just because
their prison rate is high.

Known limits: no criminal-history control (most sources lack it), plea deals
mean prosecutors shape many sentences, and the judge's own cases are part of
the baseline (small effect with hundreds of judges).
"""
import numpy as np
import pandas as pd

MIN_STRATUM = 30  # smallest class x category group used as its own baseline
MIN_CASES = 100  # judges with fewer cases in the window aren't scored
MIN_GROUP = 30  # smallest race group per judge for disparity estimates
MIN_GROUP_APPEALS = 10  # smallest number of merits decisions to state an appellate finding
# Judges whose sentences are mostly treatment-based are likely assigned to a
# specialty court (drug, mental health, veterans), so their docket isn't comparable.
SPECIALTY_TREATMENT_SHARE = 0.2
Z = 1.96


def _stratum_baseline(df: pd.DataFrame, value: pd.Series) -> pd.Series:
    """Court-wide mean of `value` for each row's comparable cases."""
    fine = value.groupby([df.offense_class, df.offense_category]).transform("mean")
    fine_n = value.groupby([df.offense_class, df.offense_category]).transform("count")
    coarse = value.groupby(df.offense_class).transform("mean")
    return fine.where(fine_n >= MIN_STRATUM, coarse).fillna(coarse)


def _excess_rate(observed: pd.Series, expected: pd.Series) -> dict:
    """Observed minus expected rate, with a 95% interval, for a set of cases."""
    n = len(observed)
    diff = observed.mean() - expected.mean()
    se = np.sqrt((expected * (1 - expected)).sum()) / n
    return {"n": n, "rate": observed.mean(), "expected": expected.mean(),
            "diff": diff, "lo": diff - Z * se, "hi": diff + Z * se}


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exp_prison"] = _stratum_baseline(df, df.prison.astype(float))
    df["exp_treatment"] = _stratum_baseline(df, df.treatment.astype(float))
    log_months = np.log(df.prison_months)
    df["log_len_resid"] = log_months - _stratum_baseline(df, log_months)
    return df


def judge_profile(cases: pd.DataFrame) -> dict:
    """Metrics for one judge's cases (rows must already have baselines)."""
    profile = {
        "judge": cases.judge.iloc[0],
        "n_cases": len(cases),
        "first_sentence": cases.sentence_date.min().date().isoformat(),
        "last_sentence": cases.sentence_date.max().date().isoformat(),
        "felony_share": cases.offense_class.isin(["M", "X", "1", "2", "3", "4"]).mean(),
        "prison": _excess_rate(cases.prison, cases.exp_prison),
        "treatment": _excess_rate(cases.treatment, cases.exp_treatment),
        "likely_specialty_court": bool(cases.treatment.mean() >= SPECIALTY_TREATMENT_SHARE),
    }

    resid = cases.log_len_resid.dropna()
    if len(resid) >= MIN_GROUP:
        se = resid.std(ddof=1) / np.sqrt(len(resid))
        # Ratio of this judge's prison terms to comparable terms court-wide (1.2 = 20% longer).
        profile["prison_length_ratio"] = {
            "n": len(resid), "ratio": float(np.exp(resid.mean())),
            "lo": float(np.exp(resid.mean() - Z * se)), "hi": float(np.exp(resid.mean() + Z * se)),
        }

    by_race = {}
    for race, group in cases.groupby("race"):
        if race != "Other/Unknown" and len(group) >= MIN_GROUP:
            by_race[race] = _excess_rate(group.prison, group.exp_prison)
    profile["prison_by_race"] = by_race

    # Within-judge gap: how much more often Black defendants go to prison than
    # white defendants facing comparable charges, in percentage points.
    if "Black" in by_race and "White" in by_race:
        b, w = by_race["Black"], by_race["White"]
        gap = b["diff"] - w["diff"]
        se = np.hypot((b["hi"] - b["lo"]) / (2 * Z), (w["hi"] - w["lo"]) / (2 * Z))
        profile["black_white_prison_gap"] = {"gap": gap, "lo": gap - Z * se, "hi": gap + Z * se}

    return profile


def all_profiles(df: pd.DataFrame, since: str, judges=None, min_cases: int = MIN_CASES) -> list[dict]:
    """Profiles for every judge (or just `judges`) with at least `min_cases`
    sentences on or after `since`.

    Baselines use every judge in the window, so judges are compared to their contemporaries.
    """
    window = add_baselines(df[df.sentence_date >= since])
    counts = window.judge.value_counts()
    eligible = counts[counts >= min_cases].index
    if judges is not None:
        eligible = eligible.intersection(list(judges))
    return [judge_profile(g) for _, g in window[window.judge.isin(eligible)].groupby("judge")]


def summary_table(profiles: list[dict]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        rows.append({
            "judge": p["judge"],
            "n_cases": p["n_cases"],
            "last_sentence": p["last_sentence"],
            "felony_share": round(p["felony_share"], 3),
            "likely_specialty_court": p["likely_specialty_court"],
            "prison_rate": round(p["prison"]["rate"], 3),
            "prison_vs_expected_pp": round(100 * p["prison"]["diff"], 1),
            "treatment_rate": round(p["treatment"]["rate"], 3),
            "treatment_vs_expected_pp": round(100 * p["treatment"]["diff"], 1),
            "prison_length_ratio": round(p.get("prison_length_ratio", {}).get("ratio", np.nan), 2),
            "black_white_gap_pp": round(100 * p.get("black_white_prison_gap", {}).get("gap", np.nan), 1),
        })
    return pd.DataFrame(rows).sort_values("prison_vs_expected_pp", ascending=False)
