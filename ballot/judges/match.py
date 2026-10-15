"""Match candidate names on a ballot to judge names in court records.

Sources disagree on middle initials, nicknames, punctuation, and hyphenated or
married surnames. A record matches when the surnames share a part ("Teresa
Molina" ~ "Teresa Molina-Gonzalez") and at least one given name matches
("Kate Moreland" ~ "Caroline Kate Moreland"; "Chris" ~ "Christopher").
"""
from difflib import SequenceMatcher
import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def name_parts(name: str) -> tuple[list[str], set[str]]:
    """(given names, surname parts), lowercase, without accents, initials, or suffixes."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’.]", "", s)
    words = [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", s) if w not in _SUFFIXES and len(w) > 1]
    if len(words) < 2:
        return [], set()
    return words[:-1], set(words[-1].split("-"))


def _same_given(a: str, b: str) -> bool:
    short, long = sorted((a, b), key=len)
    return long.startswith(short) and len(short) >= 3


def _same_surname(a: set[str], b: set[str]) -> bool:
    # Allow a one-letter typo in longer surnames ("Powelll", "Antoniett").
    return any(x == y or (min(len(x), len(y)) >= 5 and SequenceMatcher(None, x, y).ratio() >= 0.9)
               for x in a for y in b)


def matches(ballot_name: str, record_name: str) -> bool:
    b_given, b_surname = name_parts(ballot_name)
    r_given, r_surname = name_parts(record_name)
    if not _same_surname(b_surname, r_surname):
        return False
    return any(_same_given(x, y) for x in b_given for y in r_given)


def find_records(ballot_name: str, record_names) -> list[str]:
    return [n for n in record_names if matches(ballot_name, n)]
