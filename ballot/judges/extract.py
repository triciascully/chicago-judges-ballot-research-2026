"""Extract structured facts from ISBA judicial evaluations with Claude.

Claude only extracts; code derives the voter-facing flags (former prosecutor,
treatment court experience, ...). Every extracted item carries a verbatim
evidence quote, so ballot.judges.eval_extraction can check each claim against
the source text.

    python -m ballot.judges.extract --backend llama    # local llama-server (default port 8080)
    python -m ballot.judges.extract --backend claude   # needs ANTHROPIC_API_KEY

Results are cached by (model, prompt version, judge, text), so reruns only
call the model for new or changed evaluations, and each model's results are
kept apart so they can be compared.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request
PROMPT_VERSION = "3"
# Prompt history, scored against hand labels (data/eval/):
#   v3  current. Most verbatim quotes (86%); precision comes from FLAG_PATTERNS + grounding.
#   v4  stricter labeling rules: more "other"/"civil" dumping, 78% verbatim.
#   v5  recall-focused: 10/60 judges looped until the length cap, 79% verbatim.  # bump when SYSTEM or SCHEMA changes to invalidate the cache

ROLES = [
    "prosecutor", "public_defender", "private_criminal_defense", "civil_practice",
    "government_attorney", "legal_aid_or_public_interest", "judicial_staff", "academic", "other",
]
BENCH_AREAS = [
    "criminal_felony", "criminal_misdemeanor", "juvenile_justice", "domestic_violence",
    "treatment_court", "child_protection", "domestic_relations", "civil", "chancery",
    "probate", "traffic", "eviction_housing", "appellate", "administrative", "other",
]
FEEDBACK_TOPICS = [
    "temperament", "legal_knowledge", "fairness", "diligence", "courtroom_management",
    "treatment_of_litigants", "integrity", "other",
]

_EVIDENCE = {"type": "string", "description": "Exact contiguous span copied verbatim from the evaluation"}


def _list_of(properties: dict) -> dict:
    return {"type": "array", "items": {"type": "object", "properties": properties,
                                       "required": list(properties), "additionalProperties": False}}


SCHEMA = {
    "type": "object",
    "properties": {
        "current_assignment": {"type": "string", "description": "Current courtroom assignment, or empty"},
        "prior_roles": _list_of({"role": {"type": "string", "enum": ROLES}, "evidence": _EVIDENCE}),
        "bench_experience": _list_of({"area": {"type": "string", "enum": BENCH_AREAS}, "evidence": _EVIDENCE}),
        "attorney_feedback": _list_of({
            "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
            "topic": {"type": "string", "enum": FEEDBACK_TOPICS},
            "evidence": _EVIDENCE,
        }),
    },
    "required": ["current_assignment", "prior_roles", "bench_experience", "attorney_feedback"],
    "additionalProperties": False,
}

SYSTEM = """You extract facts from Illinois State Bar Association (ISBA) evaluations of Cook County judges for a voter guide. Voters will rely on this, so a missing fact is better than a wrong one.

Extract only what the evaluation states. Don't infer from names, reputations, or outside knowledge.

Each item's evidence must be copied exactly from the evaluation: one contiguous span with the same words and punctuation, as short as possible while still showing the fact. Never use "..." or skip words; a checker will search for the exact span in the text.

prior_roles: legal jobs held before becoming a judge. Assistant State's Attorney or other prosecutor -> prosecutor. Assistant Public Defender -> public_defender. Criminal defense in private practice -> private_criminal_defense.

bench_experience: every courtroom assignment as a judge, including the current one (list it here as well as in current_assignment). Mental health, drug, or veterans court calls and other problem-solving courts -> treatment_court.

attorney_feedback: what attorneys or interviewees said about the judge. Group related praise into one item (for example, a sentence listing several virtues is one item). At most 8 items. Mark criticisms negative even when the overall evaluation is favorable.

Leave a list empty when the evaluation doesn't cover it."""


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def cached_map(cache_path: Path, items: dict[str, tuple], fn, workers: int) -> dict:
    """Run fn(*args) for each {key: args} not already in the JSON cache, in parallel.

    Saves after every result, so an interrupted run resumes where it stopped.
    Items that fail are reported and left out of the returned dict.
    """
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = {k: args for k, args in items.items() if k not in cache}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fn, *args): key for key, args in todo.items()}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                cache[futures[future]] = future.result()
                cache_path.write_text(json.dumps(cache, indent=2))
            except Exception as e:  # one bad item shouldn't discard the rest
                print(f"  failed: {todo[futures[future]][1]}: {e}")
            print(f"  {done}/{len(todo)}", end="\r", flush=True)
    return {k: cache[k] for k in items if k in cache}


class Claude:
    """Claude through the Anthropic API."""
    name, workers = "claude", 8

    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def __call__(self, system: str, user: str, schema: dict) -> dict:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            # If a safety classifier declines, the API reruns the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"Claude stopped with {response.stop_reason} (request {response._request_id})")
        return json.loads(next(b.text for b in response.content if b.type == "text"))


class LlamaServer:
    """A local model served by llama.cpp's llama-server (OpenAI-compatible endpoint).
    The JSON schema is compiled to a grammar, so output always matches it."""
    name, workers = "llama", 2

    def __init__(self, url: str = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")):
        self.url = url.rstrip("/")
        with urllib.request.urlopen(f"{self.url}/v1/models", timeout=10) as r:
            self.model = json.load(r)["data"][0]["id"]

    def __call__(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 4096,  # fail fast if the model loops instead of finishing
            # llama-server's own field; some builds silently ignore OpenAI-style response_format.
            "json_schema": schema,
        }
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            choice = json.load(r)["choices"][0]
        if choice["finish_reason"] != "stop":
            raise RuntimeError(f"{self.model} stopped with {choice['finish_reason']}")
        return json.loads(choice["message"]["content"])


BACKENDS = {"claude": Claude, "llama": LlamaServer}


def extract_one(backend, name: str, text: str) -> dict:
    user = f"Judge: {name}\n\n<evaluation>\n{text}\n</evaluation>"
    return backend(SYSTEM, user, SCHEMA)


def extract_all(evaluations: dict[str, str], cache_path: Path, backend) -> dict[str, dict]:
    """{judge name: evaluation text} -> {judge name: extraction}."""
    keys = {name: _cache_key(backend.model, PROMPT_VERSION, name, text) for name, text in evaluations.items()}
    items = {keys[n]: (backend, n, t) for n, t in evaluations.items()}
    results = cached_map(cache_path, items, extract_one, workers=backend.workers)
    return {name: results[keys[name]] for name in evaluations if keys[name] in results}


# Flags turn on only when a quote contains wording that actually shows the fact. The model's
# labels decide which list a quote is in; these patterns decide whether it counts. This keeps
# the model's high recall while blocking its habit of attaching labels to unrelated quotes.
FLAG_PATTERNS = {
    # The State's Attorney's Office also has a Civil Division, which doesn't prosecute.
    "former_prosecutor": ("prior_roles", r"^(?!.*civil).*(state'?s attorney|prosecut)"),
    # Represented people accused of crimes. Civil defense (e.g. insurance defense) doesn't count.
    "former_criminal_defense": ("prior_roles", r"public defender|criminal defense|appellate defender"),
    # Worked on behalf of children, poor people, or people with disabilities.
    "represented_vulnerable_clients": ("prior_roles",
                                       r"public defender|appellate defender|public guardian|guardian ad litem"
                                       r"|child(ren)?'?s? rep|legal aid|legal assistance|indigent|low[- ]income"
                                       r"|disabilit|mental health|pro bono"),
    # The evaluation itself says they heard criminal matters. The voter-facing profile also
    # counts judges with sentencing records (see ballot.build_ballot_report).
    "criminal_bench": ("bench_experience",
                       r"criminal|felony|felonies|misdemeanor|juvenile justice|delinquen|domestic violence"
                       r"|preliminary hearing|narcotics|bond court|pretrial division|first appearance"),
    "treatment_court": ("bench_experience",
                        r"mental health|drug court|drug treatment|veteran|problem[- ]solving|treatment court"),
}


# Contrast and complaint wording. The model sometimes extracts a criticism but rates it positive
# when it sits inside a favorable paragraph ("...diligent, though some attorneys thought she moved
# cases too slowly"), so a feedback quote with this wording counts as criticism whatever its rating.
CRITICISM_PATTERN = (
    r"\b(though|although|however|despite|but)\b|concern|criticiz|complain|\btoo (slow|harsh|quick|fast)"
    r"|impatien|\bpoor (temperament|demeanor|judgment)|\black(s|ed)? of\b|\brude\b|short[- ]tempered"
    r"|\btardy\b|\bnegative\b|disrespect|unprepared|inconsistent|abrupt|condescend|arrogan"
)


def normalize(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def is_grounded(evidence: str, text: str) -> bool:
    """True if the quote appears verbatim in the text (ignoring case, quote style, spacing)."""
    quote = normalize(evidence)
    return len(quote) >= 3 and quote in normalize(text)


def flag_evidence(extraction: dict, text: str) -> dict[str, list[str]]:
    """{flag: supporting quotes}. A quote supports a flag only if it is verbatim in `text` and
    contains wording matching FLAG_PATTERNS (or CRITICISM_PATTERN), so every flag points to a
    checkable sentence."""
    def quotes(items: list[dict], keep) -> list[str]:
        return list(dict.fromkeys(i["evidence"] for i in items if is_grounded(i["evidence"], text) and keep(i)))

    evidence = {flag: quotes(extraction[field], lambda i, p=pattern: re.search(p, normalize(i["evidence"])))
                for flag, (field, pattern) in FLAG_PATTERNS.items()}
    evidence["has_criticism"] = quotes(
        extraction["attorney_feedback"],
        lambda f: f["sentiment"] != "positive" or re.search(CRITICISM_PATTERN, normalize(f["evidence"])))
    return evidence


def derive_flags(extraction: dict, text: str) -> dict[str, bool]:
    """Voter-facing yes/no flags; see flag_evidence for what counts as support."""
    return {flag: bool(quotes) for flag, quotes in flag_evidence(extraction, text).items()}


def output_path(backend_name: str) -> Path:
    from ballot.build_judge_profiles import OUT
    return OUT / f"isba_extractions_{backend_name}.json"


def main() -> None:
    from ballot.adapters import cook_bar_ratings
    from ballot.build_judge_profiles import OUT, RAW

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=BACKENDS, default="llama")
    parser.add_argument("--limit", type=int, help="only the first N judges (for a quick test)")
    args = parser.parse_args()
    backend = BACKENDS[args.backend]()

    ratings = cook_bar_ratings.load(RAW.parent / "bar_ratings")
    isba = ratings[ratings.source == "Illinois State Bar Association"].head(args.limit)
    evaluations = dict(zip(isba.name, isba.summary))

    extractions = extract_all(evaluations, OUT / "cache" / f"isba_extractions_{backend.name}.json", backend)
    rows = [{"name": n, "flags": derive_flags(e, evaluations[n]), "extraction": e} for n, e in extractions.items()]
    output_path(backend.name).write_text(json.dumps({"model": backend.model, "judges": rows}, indent=2))
    print(f"{backend.model}: {len(rows)} judges -> {output_path(backend.name)}")

    import pandas as pd
    table = pd.DataFrame([{"name": r["name"], **r["flags"]} for r in rows])
    print(table.to_string(index=False))
    print("\n" + table.drop(columns="name").sum().to_string())


if __name__ == "__main__":
    main()
