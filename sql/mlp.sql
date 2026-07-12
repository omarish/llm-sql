CREATE OR REPLACE FUNCTION mlp(
    p_layer_idx INT,
    p_x vector(768)
) RETURNS vector(768) AS $$
DECLARE
    v_out vector(768);
BEGIN
    WITH
    -- 1. Up-projection 768 -> 3072. c_fc.weight is column-major: each of the
    -- 3072 hidden units is stored as its 768-dim input-weight column at
    -- (row_idx = j, chunk_idx = 0), so the pre-activation for hidden unit j is
    -- (p_x . W[:, j]) + bias[j]. <#> is negative inner product, hence * -1.
    -- c_fc.bias is stored as 4 chunks of 768; unnest to a global index j.
    expansion AS (
        SELECT
            w.row_idx AS j,
            ((p_x <#> w.vec) * -1) + b.bias AS val
        FROM layer_weights w
        JOIN (
            SELECT (chunk_idx * 768 + (u.ord - 1)) AS j, u.bval AS bias
            FROM layer_weights,
                 UNNEST(vec::real[]) WITH ORDINALITY AS u(bval, ord)
            WHERE layer_idx = p_layer_idx AND tensor_name = 'mlp.c_fc.bias'
        ) b ON b.j = w.row_idx
        WHERE w.layer_idx = p_layer_idx AND w.tensor_name = 'mlp.c_fc.weight'
    ),
    -- 2. GELU (tanh approximation, as GPT-2 uses):
    -- 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu_applied AS (
        SELECT
            j,
            0.5 * val * (1.0 + TANH(SQRT(2.0 / PI())
                * (val + 0.044715 * POWER(val, 3)))) AS gelu_val
        FROM expansion
    ),
    -- 3. Down-projection 3072 -> 768. Assemble the hidden activations into a
    -- single vector(3072) and dot it against each output dim's precomputed wide
    -- weight vector (mlp_cproj_wide). This is 768 pgvector dot products instead
    -- of unnesting ~2.4M rows -- the big speedup. <#> is negative inner product.
    -- y[d] = sum_j gelu[j] * W_proj[j, d].
    gelu_vec AS (
        SELECT array_agg(gelu_val ORDER BY j)::vector(3072) AS v
        FROM gelu_applied
    ),
    projection AS (
        SELECT
            w.out_dim AS final_dim,
            (gv.v <#> w.vec) * -1 AS val
        FROM mlp_cproj_wide w
        CROSS JOIN gelu_vec gv
        WHERE w.layer_idx = p_layer_idx
    ),
    -- 4. Add the output bias (a single 768-dim vec at chunk_idx 0).
    final_biased AS (
        SELECT p.final_dim, p.val + b.bias AS val
        FROM projection p
        JOIN (
            SELECT (u.ord - 1) AS d, u.bval AS bias
            FROM layer_weights,
                 UNNEST(vec::real[]) WITH ORDINALITY AS u(bval, ord)
            WHERE layer_idx = p_layer_idx AND tensor_name = 'mlp.c_proj.bias'
              AND chunk_idx = 0
        ) b ON b.d = p.final_dim
    )
    SELECT array_agg(val ORDER BY final_dim)::vector(768) INTO v_out FROM final_biased;

    RETURN v_out;
END;
$$ LANGUAGE plpgsql;
