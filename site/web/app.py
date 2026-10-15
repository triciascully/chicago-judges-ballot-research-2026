"""Web app for the judges page.

Serves the static page built by `python -m ballot.build_page` (public/index.html). It is a
Flask app rather than plain static hosting so server-side features, such as a question box
backed by the RAG pipeline, can be added as routes later.

    pip install -r requirements.txt
    flask --app app run              # local: http://127.0.0.1:5000
    gunicorn app:app                 # production (Heroku runs this via the Procfile)
"""
from pathlib import Path

from flask import Flask, send_from_directory

PUBLIC = Path(__file__).parent / "public"
app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(PUBLIC, "index.html")


@app.get("/healthz")
def health():
    return {"ok": True}
