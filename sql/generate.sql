CREATE OR REPLACE FUNCTION generate_text(
    p_prompt_tokens INT[],
    p_max_new_tokens INT DEFAULT 20
) RETURNS TEXT AS $$
DECLARE
    v_session_id UUID := gen_random_uuid();
    v_current_tokens INT[] := p_prompt_tokens;
    v_next_token INT;
    v_x vector(768);
    v_attn_out vector(768);
    v_mlp_out vector(768);
    v_seq_pos INT;
    v_piece TEXT;
    v_output_text TEXT := '';
BEGIN
    -- 1. Warm up the KV Cache with the seed prompt tokens
    FOR v_seq_pos IN 0 .. (cardinality(p_prompt_tokens) - 1) LOOP
        -- Token embedding + learned positional embedding for this position
        SELECT t.vec + p.vec INTO v_x
        FROM token_embeddings t, position_embeddings p
        WHERE t.token_id = p_prompt_tokens[v_seq_pos + 1]
          AND p.position = v_seq_pos;
        
        -- Run through the 12-layer Transformer stack to populate past keys/values
        FOR i IN 0 .. 11 LOOP
            -- LayerNorm 1 + Attention + Residual Connection
            v_attn_out := compute_attention(v_session_id, v_seq_pos, i, layernorm(i, 'ln_1', v_x));
            v_x := v_x + v_attn_out;
            
            -- LayerNorm 2 + MLP + Residual Connection
            v_mlp_out := mlp(i, layernorm(i, 'ln_2', v_x));
            v_x := v_x + v_mlp_out;
        END LOOP;
    END LOOP;

    -- 2. Autoregressive Generation Loop
    v_seq_pos := cardinality(p_prompt_tokens);
    
    FOR k IN 1 .. p_max_new_tokens LOOP
        -- Token embedding + positional embedding for the most recent token
        SELECT t.vec + p.vec INTO v_x
        FROM token_embeddings t, position_embeddings p
        WHERE t.token_id = v_current_tokens[cardinality(v_current_tokens)]
          AND p.position = v_seq_pos;
        
        -- Forward pass through all layers for the single new token step
        FOR i IN 0 .. 11 LOOP
            v_attn_out := compute_attention(v_session_id, v_seq_pos, i, layernorm(i, 'ln_1', v_x));
            v_x := v_x + v_attn_out;
            
            v_mlp_out := mlp(i, layernorm(i, 'ln_2', v_x));
            v_x := v_x + v_mlp_out;
        END LOOP;
        
        -- Final Global LayerNorm before vocabulary projection
        v_x := layernorm(11, 'ln_f', v_x);
        
        -- Language Modeling Head Projection: Find the single token with highest similarity
        SELECT token_id INTO v_next_token
        FROM token_embeddings
        ORDER BY (vec <#> v_x) ASC
        LIMIT 1;
        
        -- Append the token to our sliding window state and decode it to text.
        -- GPT-2 tokens embed their own leading spaces, so no separator is added.
        v_current_tokens := array_append(v_current_tokens, v_next_token);
        SELECT token INTO v_piece FROM vocab WHERE token_id = v_next_token;
        v_output_text := v_output_text || COALESCE(v_piece, '');
        v_seq_pos := v_seq_pos + 1;
    END LOOP;

    -- Clear the session KV cache transaction log so our memory doesn't leak
    DELETE FROM kv_cache WHERE session_id = v_session_id;

    RETURN v_output_text;
END;
$$ LANGUAGE plpgsql;

-- Convenience wrapper: text in, text out (prompt + continuation) in one call.
-- End-to-end GPT-2 in a single SQL expression:  SELECT complete('Hello, I', 10);
CREATE OR REPLACE FUNCTION complete(p_prompt TEXT, p_max_new_tokens INT DEFAULT 10)
RETURNS TEXT AS $$
    SELECT p_prompt || generate_text(gpt2_encode(p_prompt), p_max_new_tokens);
$$ LANGUAGE sql;