# llm-sql

GPT-2 (small) running as a forward pass in **pure SQL** on Postgres + [pgvector](https://github.com/pgvector/pgvector).

The model weights live in regular tables as `vector` columns, and every step of the transformer — token + positional embeddings, LayerNorm, multi-head self-attention with a KV cache, the GELU MLP, the LM head, and even **BPE tokenization** — is implemented as a SQL function. You give it text and it generates text, entirely inside the database:

```sql
SELECT complete('Hello, I', 10);
```

This is a toy / learning project, not a serious inference engine. It runs at roughly **1 second per token**.

## How it works

GPT-2's weights are exported from the reference PyTorch model into CSVs, loaded into Postgres, and then a set of PL/pgSQL functions execute the forward pass using pgvector's inner-product operator (`<#>`) for the matmuls.

| Concern | Where it lives |
| --- | --- |
| Token / positional embeddings | `token_embeddings`, `position_embeddings` tables |
| Per-layer weights (attn, MLP) | `layer_weights` table (768-dim `vector` rows, chunked) |
| Precomputed MLP down-proj | `mlp_cproj_wide` table (`vector(3072)` for a fast dot product) |
| BPE tokenizer assets | `byte_encoder`, `bpe_vocab`, `bpe_merges` tables |
| Detokenizer | `vocab` table (token id → text) |
| LayerNorm | `layernorm()` |
| Self-attention (+ KV cache) | `compute_attention()`, `kv_cache` table |
| MLP (GELU) | `mlp()` |
| Text encoder (BPE) | `gpt2_encode()`, `gpt2_bpe()`, `gpt2_byte_symbols()` |
| Generation loop | `generate_text()`, `complete()` |

Key design points:
- **pgvector matmuls.** Projections are stored column-major so each output is `p_x <#> weight_column` (a dot product). `<#>` returns the negative inner product, so results are multiplied by `-1`.
- **KV cache.** `compute_attention()` caches per-head K/V vectors in `kv_cache` keyed by session, so generating token *N* is a single forward pass rather than a re-scan of the whole context. This keeps generation **linear** in sequence length instead of quadratic.
- **Tokenizer in SQL.** `gpt2_encode()` reproduces byte-level BPE (pre-tokenize → `bytes_to_unicode` → ranked merges → vocab lookup). See the caveat below about the pre-tokenization regex.

## Repository layout

```
docker-compose.yml         Postgres 18 + pgvector, tuned for local throughput
init/01-extensions.sql     Enables the `vector` extension on first boot
weights/export_pgvector.py Exports GPT-2 weights + tokenizer assets to CSVs
load.py                    Idempotent loader: schema -> COPY -> derived -> functions
sql/
  schema.sql               Tables (weights, embeddings, tokenizer, kv_cache)
  layernorm.sql            layernorm()
  attention.sql            compute_attention()
  mlp.sql                  mlp()
  encode.sql               gpt2_encode() and helpers
  generate.sql             generate_text() and complete()
demo.sh                    Encode a prompt and generate, end to end
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (for the Postgres 18 + pgvector container)
- [uv](https://github.com/astral-sh/uv) (Python env / dependency management)
- A `psql` client on your PATH (used by `load.py` and `demo.sh`)

## Quickstart

```bash
# 1. Start Postgres 18 + pgvector
docker compose up -d

# 2. Export GPT-2 weights + tokenizer assets to CSVs (downloads gpt2 once)
uv run python weights/export_pgvector.py

# 3. Load everything and install the SQL functions (idempotent)
uv run ./load.py

# 4. Generate some text
./demo.sh "Hello, I" 10
```

`demo.sh` prints the prompt, its token ids, and the generated continuation. You can also call the model directly:

```sql
-- text in, text out, in a single SQL expression
SELECT complete('The meaning of life is', 15);

-- lower-level: token ids in, continuation out
SELECT generate_text(ARRAY[15496, 11, 314], 10);

-- just the tokenizer
SELECT gpt2_encode('Hello, I');   -- {15496,11,314}
```

## Configuration

Both `load.py` and `demo.sh` honor these environment variables:

- `DATABASE_URL` — defaults to `postgresql://postgres@localhost:5432/postgres`
- `DATA_DIR` (loader only) — where the exported CSVs live; defaults to the repo root

`load.py` is idempotent: it only loads the large CSVs when a table is empty (pass `--reload` to force a re-load, e.g. after re-running the export) and re-installs every `sql/*.sql` function on each run.

## Performance

On a local machine (Postgres 18 + pgvector, GPT-2 small, greedy decode) runtime is roughly **linear in total tokens** at about **1.1–1.2 s/token**, and input vs output tokens cost about the same — the KV cache keeps generation from going quadratic. Each generated token additionally scans the 50k-row vocabulary for the argmax LM head.

The container in `docker-compose.yml` is tuned for local throughput (durability disabled, large `work_mem`/`shared_buffers`, parallelism). `load.py` also precomputes `mlp_cproj_wide` so the MLP down-projection is a handful of large pgvector dot products instead of unnesting millions of rows.

## Caveats

This is deliberately quick-and-dirty:

- **Not for production.** The Postgres container runs with `fsync=off`, `trust` authentication, and default credentials — fine for a disposable local demo, unsafe anywhere reachable.
- **GPT-2 small only**, greedy argmax decoding (deterministic, no sampling/temperature).
- **Tokenizer regex is approximate.** Postgres regex can't express GPT-2's `\p{L}`/`\p{N}` classes or its whitespace lookahead, so `gpt2_encode()` uses POSIX classes and a simplified whitespace rule. Identical to the reference tokenizer for normal single-spaced text; unusual whitespace may differ. Verify with `SELECT gpt2_encode('...')`.
- **Slow.** ~1 s/token. That's the point — it's GPT-2 in SQL.
- The exported CSVs are large and gitignored; regenerate them with the export script.

## Acknowledgements

Weights and tokenizer from OpenAI's GPT-2, via Hugging Face `transformers`. Vector storage and similarity via pgvector.
