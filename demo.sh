#!/usr/bin/env bash
# Run a text prompt through the pure-SQL GPT-2 in Postgres and print the
# prompt + continuation. Tokenization AND generation both happen in SQL
# (gpt2_encode -> generate_text), so no Python is needed at runtime.
#
# Usage: ./demo.sh [PROMPT] [NUM_NEW_TOKENS]
#   ./demo.sh                       # defaults: "Hello, I", 10 tokens
#   ./demo.sh "The meaning of life is" 15
set -euo pipefail

PROMPT="${1:-Hello, I}"
N_TOKENS="${2:-10}"
DATABASE_URL="${DATABASE_URL:-postgresql://postgres@localhost:5432/postgres}"

echo "Prompt : $PROMPT"

# The prompt is bound as a psql variable and quoted with :'prompt', so it is
# safely escaped into the SQL literals (no shell/SQL injection).
psql "$DATABASE_URL" -v prompt="$PROMPT" -v n="$N_TOKENS" -tA <<'SQL'
SELECT 'Tokens : ' || gpt2_encode(:'prompt')::text;
SELECT 'Output : ' || :'prompt' || generate_text(gpt2_encode(:'prompt'), :n);
SQL
