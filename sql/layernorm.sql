CREATE OR REPLACE FUNCTION layernorm(
    p_layer_idx INT,
    p_tensor_prefix TEXT, -- e.g., 'ln_1' or 'ln_2'
    p_vec vector(768)
) RETURNS vector(768) AS $$
DECLARE
    v_result vector(768);
BEGIN
    WITH unrolled AS (
        SELECT 
            idx,
            val,
            AVG(val) OVER () as mu,
            COALESCE(VAR_SAMP(val) OVER (), 0) as variance
        FROM UNNEST(p_vec::real[]) WITH ORDINALITY AS u(val, idx)
    ),
    gamma AS (
        SELECT idx, val 
        FROM layer_weights, UNNEST(vec::real[]) WITH ORDINALITY AS u(val, idx)
        WHERE layer_idx = p_layer_idx AND tensor_name = p_tensor_prefix || '.weight' AND row_idx = 0
    ),
    beta AS (
        SELECT idx, val 
        FROM layer_weights, UNNEST(vec::real[]) WITH ORDINALITY AS u(val, idx)
        WHERE layer_idx = p_layer_idx AND tensor_name = p_tensor_prefix || '.bias' AND row_idx = 0
    )
    SELECT array_agg(((u.val - u.mu) / SQRT(u.variance + 1e-5) * g.val) + b.val ORDER BY u.idx)::vector(768)
    INTO v_result
    FROM unrolled u
    JOIN gamma g ON u.idx = g.idx
    JOIN beta b ON u.idx = b.idx;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql STABLE;