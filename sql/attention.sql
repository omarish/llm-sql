-- Single-token self-attention for one transformer layer.
CREATE OR REPLACE FUNCTION compute_attention(
    p_session_id UUID,
    p_seq_pos INT,
    p_layer_idx INT,
    p_x vector(768)
) RETURNS vector(768) AS $$
DECLARE
    v_out vector(768);
BEGIN
    -- 1. Project Q/K/V, add biases, split into 12 heads of 64, and cache all
    --    three. This runs as its OWN statement (not a data-modifying CTE) so
    --    the inserted rows are visible to the attention read below: Postgres
    --    does NOT expose a WITH-INSERT's rows to sibling CTEs, which would
    --    otherwise leave the current token with no K/V to attend to and yield
    --    a NULL context ("array must not contain nulls").
    INSERT INTO kv_cache (session_id, seq_pos, layer_idx, tensor_name, head_idx, vec)
    WITH qkv_proj AS (
        SELECT
            chunk_idx AS qkv_type,   -- 0=Q, 1=K, 2=V
            row_idx,
            ((p_x <#> vec) * -1) AS val
        FROM layer_weights
        WHERE layer_idx = p_layer_idx AND tensor_name = 'attn.c_attn.weight'
    ),
    qkv_biased AS (
        SELECT p.qkv_type, p.row_idx, p.val + b.val AS val
        FROM qkv_proj p
        JOIN (
            -- c_attn.bias is 3 chunks (Q/K/V) of a 768-dim vec; unnest to one
            -- scalar per (qkv_type, row_idx).
            SELECT chunk_idx AS qkv_type, (u.ord - 1) AS row_idx, u.val
            FROM layer_weights,
                 UNNEST(vec::real[]) WITH ORDINALITY AS u(val, ord)
            WHERE layer_idx = p_layer_idx AND tensor_name = 'attn.c_attn.bias'
        ) b ON p.qkv_type = b.qkv_type AND p.row_idx = b.row_idx
    ),
    heads AS (
        SELECT qkv_type, (row_idx / 64) AS head_idx,
               array_agg(val ORDER BY row_idx)::vector(64) AS head_vec
        FROM qkv_biased
        GROUP BY qkv_type, (row_idx / 64)
    )
    SELECT p_session_id, p_seq_pos, p_layer_idx,
           CASE qkv_type WHEN 0 THEN 'Q' WHEN 1 THEN 'K' ELSE 'V' END,
           head_idx, head_vec
    FROM heads;

    -- 2. Self-attention: this position's Q against all cached K (causal),
    --    softmax, weighted sum of V, concat heads, output projection + bias.
    WITH attention_scores AS (
        SELECT q.head_idx, k.seq_pos AS cache_pos,
               ((q.vec <#> k.vec) * -1) / 8.0 AS raw_score   -- scale by 1/sqrt(64)
        FROM kv_cache q
        JOIN kv_cache k ON k.session_id = p_session_id
                       AND k.layer_idx = p_layer_idx
                       AND k.tensor_name = 'K'
                       AND k.head_idx = q.head_idx
                       AND k.seq_pos <= p_seq_pos            -- causal mask
        WHERE q.session_id = p_session_id
          AND q.layer_idx = p_layer_idx
          AND q.tensor_name = 'Q'
          AND q.seq_pos = p_seq_pos
    ),
    softmax AS (
        SELECT head_idx, cache_pos,
               EXP(raw_score) / SUM(EXP(raw_score)) OVER (PARTITION BY head_idx) AS prob
        FROM attention_scores
    ),
    -- Weighted sum of V per (head, dim); split in two so SUM() isn't nested
    -- inside array_agg().
    weighted_elems AS (
        SELECT s.head_idx, v_val.dim_idx, SUM(s.prob * v_val.val) AS val
        FROM softmax s
        JOIN kv_cache v ON v.session_id = p_session_id
                       AND v.layer_idx = p_layer_idx
                       AND v.tensor_name = 'V'
                       AND v.head_idx = s.head_idx
                       AND v.seq_pos = s.cache_pos
        CROSS JOIN LATERAL UNNEST(v.vec::real[]) WITH ORDINALITY AS v_val(val, dim_idx)
        GROUP BY s.head_idx, v_val.dim_idx
    ),
    weighted_values AS (
        SELECT head_idx, array_agg(val ORDER BY dim_idx)::vector(64) AS context_head
        FROM weighted_elems
        GROUP BY head_idx
    ),
    -- Concatenate the 12 heads back into a 768-dim vector.
    concatenated AS (
        SELECT array_agg(val ORDER BY head_idx, dim_idx)::vector(768) AS context_vec
        FROM weighted_values
        CROSS JOIN LATERAL UNNEST(context_head::real[]) WITH ORDINALITY AS u(val, dim_idx)
    ),
    -- Output projection c_proj (column-major) + bias.
    projected AS (
        SELECT array_agg(((c.context_vec <#> w.vec) * -1) + b.val ORDER BY w.row_idx)::vector(768) AS final_vec
        FROM concatenated c
        CROSS JOIN layer_weights w
        JOIN (
            SELECT (u.ord - 1) AS row_idx, u.val
            FROM layer_weights,
                 UNNEST(vec::real[]) WITH ORDINALITY AS u(val, ord)
            WHERE layer_idx = p_layer_idx AND tensor_name = 'attn.c_proj.bias' AND chunk_idx = 0
        ) b ON w.row_idx = b.row_idx
        WHERE w.layer_idx = p_layer_idx AND w.tensor_name = 'attn.c_proj.weight' AND w.chunk_idx = 0
    )
    SELECT final_vec INTO v_out FROM projected;

    -- 3. Drop the transient Q rows for this position; K/V stay cached for
    --    future tokens (generate_text() clears the whole session at the end).
    DELETE FROM kv_cache
    WHERE session_id = p_session_id AND layer_idx = p_layer_idx
      AND tensor_name = 'Q' AND seq_pos = p_seq_pos;

    RETURN v_out;
END;
$$ LANGUAGE plpgsql;
