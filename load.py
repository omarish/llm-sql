#!/usr/bin/env python3
"""Idempotent loader for the GPT-2 pgvector demo.

Running this repeatedly is cheap and safe:

  * Schema is applied every run (CREATE ... IF NOT EXISTS).
  * The big CSVs are loaded ONLY when their table is empty. Pass --reload to
    force a TRUNCATE + reload (e.g. after re-running export_pgvector.py).
  * Every sql/*.sql function file is (re)installed every run, since they are
    CREATE OR REPLACE and cheap. Drop a new .sql file in sql/ and it gets
    picked up automatically -- no need to touch this script.

Config via env vars:
  DATABASE_URL  (default: postgresql://postgres@localhost:5432/postgres)
  DATA_DIR      directory holding the *_vector.csv files (default: repo root)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"
SCHEMA_FILE = SQL_DIR / "schema.sql"

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT))
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres@localhost:5432/postgres"
)

# (table name, CSV filename). Column order matches the CSVs from
# export_pgvector.py so \copy needs no explicit column list.
CSV_TABLES = [
    ("token_embeddings", "token_embeddings_vector.csv"),
    ("position_embeddings", "position_embeddings_vector.csv"),
    ("layer_weights", "layer_weights_vector.csv"),
    ("vocab", "vocab_vector.csv"),
    ("byte_encoder", "byte_encoder.csv"),
    ("bpe_vocab", "bpe_vocab.csv"),
    ("bpe_merges", "bpe_merges.csv"),
]


def psql(*args: str, capture: bool = False) -> str:
    """Run psql with ON_ERROR_STOP; exit the script on any failure."""
    cmd = ["psql", DATABASE_URL, "-v", "ON_ERROR_STOP=1", *args]
    result = subprocess.run(cmd, text=True, capture_output=capture)
    if result.returncode != 0:
        if capture and result.stderr:
            sys.stderr.write(result.stderr)
        sys.exit(f"psql failed ({result.returncode}): {' '.join(args)}")
    return result.stdout if capture else ""


def scalar(sql: str) -> str:
    return psql("-tAc", sql, capture=True).strip()


def row_count(table: str) -> int:
    # to_regclass returns NULL (empty string via -tA) if the table is absent.
    if scalar(f"SELECT to_regclass('{table}') IS NOT NULL;") != "t":
        return 0
    return int(scalar(f"SELECT count(*) FROM {table};"))


def copy_csv(table: str, csv_name: str) -> None:
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        sys.exit(f"Missing CSV for {table}: {csv_path}")
    psql("-c", f"\\copy {table} FROM '{csv_path}' WITH CSV;")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Force TRUNCATE + reload of CSV data even if already present.",
    )
    args = parser.parse_args()

    print("==> Applying schema")
    psql("-f", str(SCHEMA_FILE))

    loaded = set()
    for table, csv_name in CSV_TABLES:
        count = row_count(table)
        if args.reload:
            print(f"==> Reloading {table} (--reload)")
            psql("-c", f"TRUNCATE {table};")
            copy_csv(table, csv_name)
            loaded.add(table)
        elif count == 0:
            print(f"==> Loading {table} (currently empty)")
            copy_csv(table, csv_name)
            loaded.add(table)
        else:
            print(f"==> Skipping {table} ({count} rows already loaded)")

    print("==> Indexing + analyzing")
    psql(
        "-c",
        "CREATE INDEX IF NOT EXISTS idx_layer_weights_lookup "
        "ON layer_weights (layer_idx, tensor_name, row_idx);",
    )
    psql(
        "-c",
        "ANALYZE token_embeddings; ANALYZE position_embeddings; "
        "ANALYZE layer_weights; ANALYZE vocab; ANALYZE bpe_merges;",
    )

    # Rebuild the precomputed wide MLP projection weights whenever layer_weights
    # was (re)loaded or the derived table is empty. Assembles each output dim's
    # 4x768 chunks into a single vector(3072) ordered by global input index j.
    if "layer_weights" in loaded or row_count("mlp_cproj_wide") == 0:
        print("==> Building mlp_cproj_wide")
        psql("-c", "TRUNCATE mlp_cproj_wide;")
        psql(
            "-c",
            "INSERT INTO mlp_cproj_wide (layer_idx, out_dim, vec) "
            "SELECT layer_idx, row_idx, "
            "       array_agg(elem ORDER BY chunk_idx, ord)::vector(3072) "
            "FROM layer_weights, "
            "     LATERAL unnest(vec::real[]) WITH ORDINALITY AS u(elem, ord) "
            "WHERE tensor_name = 'mlp.c_proj.weight' "
            "GROUP BY layer_idx, row_idx;",
        )
        psql("-c", "ANALYZE mlp_cproj_wide;")

    print("==> Installing functions")
    function_files = sorted(p for p in SQL_DIR.glob("*.sql") if p != SCHEMA_FILE)
    for path in function_files:
        print(f"    - {path.name}")
        psql("-f", str(path))

    print("==> Done.")


if __name__ == "__main__":
    main()
