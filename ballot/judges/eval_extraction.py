"""Evaluate the ISBA extractions on three levels.

1. Grounding (free, deterministic): does every evidence quote actually appear
   in the evaluation text? Catches invented or paraphrased evidence.
2. Support (LLM judge): does each quote actually establish the claim made
   about it? Catches real quotes attached to the wrong label, e.g. "assistant
   public defender" tagged as prosecutor.
3. Accuracy against hand labels: precision/recall of the derived flags on
   judges you labeled yourself. The only check that catches facts the
   extractor missed entirely.

    python -m ballot.judges.eval_extraction --make-gold                 # write a labeling sheet
    python -m ballot.judges.eval_extraction --make-holdout              # a second sheet of different judges
    python -m ballot.judges.eval_extraction --backend llama --no-judge  # free checks only
    python -m ballot.judges.eval_extraction --backend llama             # judged by Claude

The support judge should be at least as capable as the extractor, and ideally a
different model; a model grading its own output tends to agree with itself.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ballot.build_judge_profiles import OUT
from ballot.judges.extract import BACKENDS, _cache_key, cached_map, derive_flags, is_grounded, output_path

GOLD = OUT.parent / "eval" / "isba_gold.csv"  # tuning set: labels the prompt was adjusted against
HOLDOUT = OUT.parent / "eval" / "isba_holdout.csv"  # held out: never used for tuning, so scores stay honest
FLAGS = ["former_prosecutor", "former_criminal_defense", "represented_vulnerable_clients",
         "criminal_bench", "treatment_court", "has_criticism"]
RENAMED = {"former_defense": "former_criminal_defense"}  # older label sheets


def claims(extraction: dict) -> list[dict]:
    """Flatten an extraction into individually checkable (claim, evidence) pairs."""
    out = [{"claim": f"Before becoming a judge, worked as: {r['role']}", "evidence": r["evidence"]}
           for r in extraction["prior_roles"]]
    out += [{"claim": f"Has served as a judge in: {b['area']}", "evidence": b["evidence"]}
            for b in extraction["bench_experience"]]
    out += [{"claim": f"Attorney feedback that is {f['sentiment']} about {f['topic']}", "evidence": f["evidence"]}
            for f in extraction["attorney_feedback"]]
    return out


def grounding_failures(extraction: dict, text: str) -> list[dict]:
    return [c for c in claims(extraction) if not is_grounded(c["evidence"], text)]


def sentencing_check(rows: list[dict]) -> pd.DataFrame:
    """Judges with a full sentencing profile (100+ sentences) have certainly sat on a
    criminal bench, which gives a model-independent recall check for criminal_bench."""
    from ballot.judges.match import matches
    report = json.loads((OUT / "cook_2026_retention_judges.json").read_text())
    criminal = [j["name"] for j in report["judges"] if j["status"] == "profile"]
    out = []
    for name in criminal:
        row = next((r for r in rows if matches(name, r["name"])), None)
        out.append({"judge": name, "criminal_bench": row["flags"]["criminal_bench"] if row else None})
    return pd.DataFrame(out)


JUDGE_SYSTEM = """You audit facts extracted from a bar association evaluation of a judge. For each numbered claim, decide whether its evidence quote, read in the context of the full evaluation, establishes the claim.

supported: a careful reader would agree the evaluation says this.
unsupported: the evidence is about something else, the label is wrong, or the claim goes beyond what the text says.

Be strict. Wrong facts in a voter guide are worse than missing ones."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim_number": {"type": "integer"},
                       "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
                       "reason": {"type": "string"}},
        "required": ["claim_number", "verdict", "reason"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False,
}


def judge_support(backend, name: str, text: str, extraction: dict) -> list[dict]:
    numbered = "\n".join(f"{i}. {c['claim']}\n   Evidence: \"{c['evidence']}\""
                         for i, c in enumerate(claims(extraction), 1))
    if not numbered:
        return []
    user = f"Judge: {name}\n\n<evaluation>\n{text}\n</evaluation>\n\n<claims>\n{numbered}\n</claims>"
    return backend(JUDGE_SYSTEM, user, JUDGE_SCHEMA)["verdicts"]


def make_label_sheet(path, rows: list[dict], texts: dict[str, str], exclude=(), n: int = 15) -> None:
    """A sheet to label by hand from the evaluation text alone, without looking at the extractions."""
    for existing in (path, path.with_suffix(".numbers")):
        if existing.exists():
            raise FileExistsError(f"{existing} already exists; refusing to overwrite your labels")
    names = pd.Series([r["name"] for r in rows if r["name"] not in set(exclude)])
    sample = names.sample(min(n, len(names)), random_state=0)
    sheet = pd.DataFrame({"name": sample, **{f: "" for f in FLAGS}, "evaluation": sample.map(texts)})
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False)
    print(f"Wrote {path}. Fill each flag column with 1 or 0 using only the evaluation text.")


def load_labels(path) -> pd.DataFrame | None:
    """Hand labels from a .csv sheet, or from the .numbers copy if it was saved more recently
    (Numbers saves in its own format by default). Rows with any blank flag are skipped."""
    numbers = path.with_suffix(".numbers")
    if numbers.exists() and (not path.exists() or numbers.stat().st_mtime > path.stat().st_mtime):
        from numbers_parser import Document  # pip install numbers-parser
        rows = Document(numbers).sheets[0].tables[0].rows(values_only=True)
        gold = pd.DataFrame(rows[1:], columns=rows[0])
    elif path.exists():
        gold = pd.read_csv(path)
    else:
        return None
    gold = gold.rename(columns=RENAMED)
    present = [f for f in FLAGS if f in gold.columns]
    bad = [(r["name"], f, r[f]) for _, r in gold.iterrows() for f in present
           if pd.notna(r[f]) and r[f] not in (0, 1)]
    if bad:
        raise ValueError(f"Labels must be 0 or 1; fix these cells: {bad}")
    return gold if gold[present].notna().any(axis=None) else None


def gold_scores(rows: list[dict], gold: pd.DataFrame) -> pd.DataFrame:
    """Per-flag agreement. Flags missing from the sheet, blank cells, and judges the model
    failed on are skipped, so a partly labeled sheet still scores what it can."""
    predicted = {r["name"]: r["flags"] for r in rows}
    out = []
    for flag in [f for f in FLAGS if f in gold.columns]:
        labeled = gold[gold[flag].notna() & gold.name.isin(predicted)]
        if labeled.empty:
            continue
        truth = labeled[flag].astype(int).astype(bool)
        pred = labeled.name.map(lambda n: predicted[n][flag])
        tp, fp, fn = (truth & pred).sum(), (~truth & pred).sum(), (truth & ~pred).sum()
        out.append({"flag": flag, "labeled": len(labeled), "agree": int((truth == pred).sum()),
                    "false_pos": int(fp), "false_neg": int(fn),
                    "precision": round(tp / (tp + fp), 2) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 2) if tp + fn else None,
                    "model_said_yes_you_said_no": labeled.name[~truth & pred].tolist(),
                    "model_said_no_you_said_yes": labeled.name[truth & ~pred].tolist()})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-gold", action="store_true", help="write a hand-labeling sheet and exit")
    parser.add_argument("--make-holdout", action="store_true",
                        help="write a second labeling sheet of judges not in the first, and exit")
    parser.add_argument("--backend", choices=BACKENDS, default="llama", help="whose extractions to score")
    parser.add_argument("--file", help="score this saved extraction file instead (e.g. an earlier version)")
    parser.add_argument("--judge", choices=BACKENDS, default="claude", help="model for the support check")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM support check")
    args = parser.parse_args()

    from ballot.adapters import cook_bar_ratings
    from ballot.build_judge_profiles import RAW
    ratings = cook_bar_ratings.load(RAW.parent / "bar_ratings")
    isba = ratings[ratings.source == "Illinois State Bar Association"]
    texts = dict(zip(isba.name, isba.summary))
    extracted = json.loads(Path(args.file or output_path(args.backend)).read_text())
    # Recompute flags so changes to derive_flags apply to saved extractions without rerunning the model.
    rows = [{**r, "flags": derive_flags(r["extraction"], texts[r["name"]])} for r in extracted["judges"]]
    print(f"Scoring {extracted['model']} extractions of {len(rows)} judges\n")

    if args.make_gold:
        make_label_sheet(GOLD, rows, texts)
        return
    if args.make_holdout:
        tuned = load_labels(GOLD)
        make_label_sheet(HOLDOUT, rows, texts, exclude=[] if tuned is None else tuned.name)
        return

    report = {}
    n_claims = sum(len(claims(r["extraction"])) for r in rows)
    ungrounded = {r["name"]: f for r in rows if (f := grounding_failures(r["extraction"], texts[r["name"]]))}
    n_bad = sum(len(f) for f in ungrounded.values())
    print(f"1. Grounding: {n_claims - n_bad}/{n_claims} evidence quotes found verbatim in the source")
    for name, failures in ungrounded.items():
        for f in failures:
            print(f"   {name}: {f['claim']} <- \"{f['evidence']}\"")
    report["grounding"] = {"claims": n_claims, "ungrounded": ungrounded}

    if not args.no_judge:
        judge = BACKENDS[args.judge]()
        if judge.model == extracted["model"]:
            print("Note: the same model is extracting and judging, so the support check is weak.\n")
        items = {_cache_key(judge.model, "judge-1", r["name"], json.dumps(r["extraction"], sort_keys=True)):
                 (judge, r["name"], texts[r["name"]], r["extraction"]) for r in rows}
        verdicts = cached_map(OUT / "cache" / f"isba_support_{args.judge}.json", items, judge_support,
                              workers=judge.workers)
        unsupported = []
        for key, r in zip(items, rows):
            judged = claims(r["extraction"])
            unsupported += [{"name": r["name"], **judged[x["claim_number"] - 1], "reason": x["reason"]}
                            for x in verdicts.get(key, []) if x["verdict"] == "unsupported" and 0 < x["claim_number"] <= len(judged)]
        print(f"\n2. Support ({judge.model}): {n_claims - len(unsupported)}/{n_claims} claims judged supported")
        for u in unsupported:
            print(f"   {u['name']}: {u['claim']}\n      evidence: \"{u['evidence']}\"\n      judge: {u['reason']}")
        report["support"] = {"unsupported": unsupported}

    check = sentencing_check(rows)
    found = check.criminal_bench.fillna(False).astype(bool)
    print(f"\n4. Sentencing cross-check: criminal_bench set for {found.sum()}/{len(check)} judges "
          f"with 100+ sentences in court records")
    for name in check.judge[~found]:
        print(f"   missed: {name}")
    report["sentencing_check"] = check.to_dict("records")

    report["labels"] = {}
    for title, path in [("tuning set", GOLD), ("held-out set", HOLDOUT)]:
        gold = load_labels(path)
        if gold is None:
            print(f"\n3. Accuracy vs your labels, {title}: no labels yet ({path.name})")
            continue
        scores = gold_scores(rows, gold)
        print(f"\n3. Accuracy vs your labels, {title} ({len(gold)} judges):")
        print(scores.drop(columns=["model_said_yes_you_said_no", "model_said_no_you_said_yes"]).to_string(index=False))
        for s in scores.itertuples():
            for who, names in [("model yes / you no", s.model_said_yes_you_said_no),
                               ("model no / you yes", s.model_said_no_you_said_yes)]:
                if names:
                    print(f"   {s.flag}, {who}: {', '.join(names)}")
        report["labels"][title] = scores.to_dict("records")

    (OUT / f"isba_extraction_eval_{args.backend}.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
