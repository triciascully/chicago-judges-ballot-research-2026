# Cook County judicial retention research

A voter guide to the judges on the Cook County retention ballot, and a hands-on project in
retrieval, structured extraction, and evaluating LLM output.

- `ballot/adapters/`: source-specific loaders (Cook County sentencing data, bar ratings)
- `ballot/judges/`: jurisdiction-agnostic metrics, name matching, ISBA extraction and its evaluation, profiles
- `data/eval/`: hand labels used to score the extraction (`LABELING.md` defines each flag)
- `site/`: the shareable page (`template.html`) and a Flask app for self-hosting (`site/web/`)
- `foia/`: records requests and outreach drafts

## Rebuild

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ballot.build_ballot_report   # downloads source data on first run
.venv/bin/python -m ballot.build_page            # writes site/judges.html and site/web/public/index.html
```

The ISBA extraction runs on a local model (`--backend llama`, llama.cpp's llama-server) or
Claude (`--backend claude`, needs `ANTHROPIC_API_KEY`):

```
.venv/bin/python -m ballot.judges.extract --backend llama
.venv/bin/python -m ballot.judges.eval_extraction --backend llama --no-judge
```
