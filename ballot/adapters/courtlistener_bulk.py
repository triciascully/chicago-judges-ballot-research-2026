"""Adapter: CourtListener bulk data -> Illinois and Michigan appellate decisions, as Parquet.

The API allows 125 requests a day on a free account, so this reads Free Law Project's
quarterly database dumps instead (https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/,
snapshot 2026-09-30). The dumps cover every court, so each one is streamed through a parallel
bzip2 decoder and filtered on the fly; only the courts in COURTS are kept (Illinois for the
Cook County guide, Michigan for the Detroit guide), tagged by court:

    dockets (5 GB)    -> cases in those courts             -> dockets.parquet
    clusters (2.5 GB) -> their decisions since 2015        -> clusters.parquet
    opinions (55 GB)  -> those decisions' text             -> opinions.parquet

    python -m ballot.adapters.courtlistener_bulk dockets|clusters|opinions|all

Download the dumps first (data/raw/courtlistener/bulk/, e.g. with aria2c); they can be
deleted once the Parquet files exist.
"""
import csv
import io
import os
from pathlib import Path
import re
import sys
import time

import indexed_bzip2
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
BULK = ROOT / "data" / "raw" / "courtlistener" / "bulk"
OUT = ROOT / "data" / "processed" / "courtlistener"
SNAPSHOT = "2026-09-30"
COURTS = {
    "illappct": "Appellate Court of Illinois",  # reviews Cook County judges
    "ill": "Illinois Supreme Court",
    "michctapp": "Michigan Court of Appeals",    # reviews Wayne County judges
    "mich": "Michigan Supreme Court",            # also decides judicial discipline
}
SINCE = "2015-01-01"
BATCH = 2000
csv.field_size_limit(sys.maxsize)


def rows(name: str, columns: list[str]):
    """Yield the chosen columns of each row of a bulk CSV, decompressing on every core.
    The dumps are Postgres CSV with backslash escapes (see Free Law Project's load script)."""
    path = BULK / f"{name}-{SNAPSHOT}.csv.bz2"
    if (path.parent / (path.name + ".aria2")).exists():
        raise RuntimeError(f"{path.name} is still downloading")
    with indexed_bzip2.open(str(path), parallelization=os.cpu_count()) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text, escapechar="\\", doublequote=False)
        header = next(reader)
        idx = [header.index(c) for c in columns]
        width = len(header)
        for r in reader:
            if len(r) == width:
                yield [r[i] for i in idx]


def _write(path: Path, columns: list[str], keep, source) -> int:
    """Stream rows that pass `keep` into a zstd-compressed Parquet file; returns rows kept."""
    OUT.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([(c, pa.string()) for c in columns])
    tmp, kept, seen, start = path.with_suffix(".part"), 0, 0, time.time()
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        batch = []
        for r in source:
            seen += 1
            if keep(r):
                batch.append(r)
                kept += 1
                if len(batch) >= BATCH:
                    writer.write_table(pa.Table.from_pylist([dict(zip(columns, b)) for b in batch], schema))
                    batch = []
            if seen % 1_000_000 == 0:
                print(f"  {path.stem}: {seen:,} rows read, {kept:,} kept ({time.time() - start:.0f} s)", flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist([dict(zip(columns, b)) for b in batch], schema))
    tmp.rename(path)
    print(f"{path.name}: kept {kept:,} of {seen:,} rows in {time.time() - start:.0f} s", flush=True)
    return kept


def dockets() -> None:
    cols = ["id", "court_id", "docket_number", "case_name", "appeal_from_str", "date_filed"]
    _write(OUT / "dockets.parquet", cols, lambda r: r[1] in COURTS, rows("dockets", cols))


def clusters() -> None:
    d = pq.read_table(OUT / "dockets.parquet", columns=["id", "court_id"]).to_pydict()
    court_of = dict(zip(d["id"], d["court_id"]))
    cols = ["id", "docket_id", "date_filed", "case_name", "precedential_status", "disposition", "judges"]

    def tagged(source):  # add each decision's court, from its docket
        for r in source:
            yield r + [court_of.get(r[1], "")]

    _write(OUT / "clusters.parquet", cols + ["court_id"], lambda r: r[-1] in COURTS and r[2] >= SINCE,
           tagged(rows("opinion-clusters", cols)))


def _plain(html: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", " ", html))


def opinions() -> None:
    ids = set(pq.read_table(OUT / "clusters.parquet", columns=["id"]).column("id").to_pylist())
    src = ["id", "cluster_id", "type", "plain_text", "html_with_citations", "html"]
    out = ["id", "cluster_id", "type", "text"]

    def slim(source):
        for r in source:
            if r[1] in ids:
                yield [r[0], r[1], r[2], r[3] or _plain(r[4] or r[5])]
            else:
                yield None  # still counted, so progress shows rows read

    _write(OUT / "opinions.parquet", out, lambda r: r is not None, slim(rows("opinions", src)))


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("dockets", dockets), ("clusters", clusters), ("opinions", opinions)):
        if step in (name, "all"):
            fn()


if __name__ == "__main__":
    main()
