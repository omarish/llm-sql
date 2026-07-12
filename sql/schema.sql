-- Schema for the GPT-2 pgvector export.
-- Column order matches the CSVs produced by weights/export_pgvector.py so the
-- \copy commands in load.py work without an explicit column list.

CREATE EXTENSION IF NOT EXISTS vector;

-- token_embeddings_vector.csv rows: (token_id, vec)
CREATE TABLE IF NOT EXISTS token_embeddings (
    token_id INT PRIMARY KEY,
    vec      vector(768) NOT NULL
);

-- position_embeddings_vector.csv rows: (position, vec). GPT-2 (wpe) has 1024
-- learned positional vectors; generate.sql adds these to the token embedding.
CREATE TABLE IF NOT EXISTS position_embeddings (
    position INT PRIMARY KEY,
    vec      vector(768) NOT NULL
);

-- layer_weights_vector.csv rows: (layer_idx, tensor_name, row_idx, chunk_idx, vec)
-- export_pgvector.py chunks every weight into 768-wide pieces; chunk_idx
-- disambiguates pieces of the same row. For attn.c_attn.weight/bias, chunk_idx
-- also doubles as the Q/K/V block (0=Q, 1=K, 2=V). vec is uniformly 768-dim.
CREATE TABLE IF NOT EXISTS layer_weights (
    layer_idx   INT  NOT NULL,
    tensor_name TEXT NOT NULL,
    row_idx     INT  NOT NULL,
    chunk_idx   INT  NOT NULL,
    vec         vector(768) NOT NULL
);

-- Migrate tables created before the column was renamed chunk -> chunk_idx.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'layer_weights' AND column_name = 'chunk'
    ) THEN
        ALTER TABLE layer_weights RENAME COLUMN chunk TO chunk_idx;
    END IF;
END $$;

-- If the table exists from an earlier run with a different vec dimension
-- (e.g. unconstrained vector), pin it back to 768. Guarded so re-runs don't
-- rewrite/validate an already-correct populated table. pgvector stores the
-- dimension directly in atttypmod (-1 = unconstrained).
DO $$
BEGIN
    IF (SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'layer_weights'::regclass AND attname = 'vec')
       IS DISTINCT FROM 768 THEN
        ALTER TABLE layer_weights ALTER COLUMN vec TYPE vector(768);
    END IF;
END $$;

-- vocab_vector.csv rows: (token_id, token). Maps token ids back to text so
-- generate_text() can return readable output.
CREATE TABLE IF NOT EXISTS vocab (
    token_id INT PRIMARY KEY,
    token    TEXT NOT NULL
);

-- Precomputed wide form of mlp.c_proj.weight: one vector(3072) per output dim
-- (assembled from the 4 chunks in layer_weights by load.py). Lets mlp() do the
-- down-projection as 768 pgvector dot products instead of unnesting ~2.4M rows
-- per layer -- the main forward-pass speedup.
CREATE TABLE IF NOT EXISTS mlp_cproj_wide (
    layer_idx INT NOT NULL,
    out_dim   INT NOT NULL,
    vec       vector(3072) NOT NULL,
    PRIMARY KEY (layer_idx, out_dim)
);

-- BPE encoder assets used by gpt2_encode() to tokenize text into token ids.
-- byte_encoder: byte value (0..255) -> unicode char (GPT-2 bytes_to_unicode).
CREATE TABLE IF NOT EXISTS byte_encoder (
    byte  INT PRIMARY KEY,
    uchar TEXT NOT NULL
);

-- bpe_vocab: byte-char token string (e.g. 'Ġthe') -> token id (encoder.json).
CREATE TABLE IF NOT EXISTS bpe_vocab (
    token    TEXT PRIMARY KEY,
    token_id INT NOT NULL
);

-- bpe_merges: ranked merge rules (merges.txt); lower rank merges first.
CREATE TABLE IF NOT EXISTS bpe_merges (
    rank       INT PRIMARY KEY,
    pair_left  TEXT NOT NULL,
    pair_right TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bpe_merges_pair
    ON bpe_merges (pair_left, pair_right);

-- Per-session KV cache written by compute_attention() in attention.sql.
-- Runtime/session state (not loaded from CSV), so load.py never truncates it.
-- Each row is one head's 64-dim K or V vector at a given sequence position.
CREATE TABLE IF NOT EXISTS kv_cache (
    session_id  UUID NOT NULL,
    seq_pos     INT  NOT NULL,
    layer_idx   INT  NOT NULL,
    tensor_name TEXT NOT NULL,   -- 'K' or 'V'
    head_idx    INT  NOT NULL,
    vec         vector(64) NOT NULL,
    PRIMARY KEY (session_id, layer_idx, tensor_name, head_idx, seq_pos)
);
