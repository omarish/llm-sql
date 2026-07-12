-- Pure-SQL GPT-2 BPE encoder: text -> token id array, mirroring the reference
-- tokenizer (byte-level pre-tokenization -> bytes_to_unicode -> ranked merges
-- -> vocab lookup).

-- Map a pre-token's UTF-8 bytes to the GPT-2 byte-level unicode symbols, one
-- array element per byte. These are the starting symbols for the merge loop.
CREATE OR REPLACE FUNCTION gpt2_byte_symbols(p_chunk TEXT)
RETURNS TEXT[] AS $func$
DECLARE
    b bytea := convert_to(p_chunk, 'UTF8');
    res TEXT[] := ARRAY[]::TEXT[];
    i int;
BEGIN
    FOR i IN 0 .. octet_length(b) - 1 LOOP
        res := res || (SELECT uchar FROM byte_encoder WHERE byte = get_byte(b, i));
    END LOOP;
    RETURN res;
END;
$func$ LANGUAGE plpgsql STABLE;

-- Apply the BPE merge loop to a list of symbols: repeatedly merge every
-- occurrence of the lowest-rank adjacent pair until no ranked pair remains.
CREATE OR REPLACE FUNCTION gpt2_bpe(p_symbols TEXT[])
RETURNS TEXT[] AS $func$
DECLARE
    w TEXT[] := p_symbols;
    best_left TEXT;
    best_right TEXT;
    best_rank int;
    nw TEXT[];
    i int;
    n int;
BEGIN
    IF array_length(w, 1) IS NULL OR array_length(w, 1) < 2 THEN
        RETURN w;
    END IF;

    LOOP
        -- Lowest-rank adjacent pair currently in the word.
        SELECT m.pair_left, m.pair_right, m.rank
        INTO best_left, best_right, best_rank
        FROM (
            SELECT w[g] AS l, w[g + 1] AS r
            FROM generate_subscripts(w, 1) AS g
            WHERE g < array_length(w, 1)
        ) p
        JOIN bpe_merges m ON m.pair_left = p.l AND m.pair_right = p.r
        ORDER BY m.rank
        LIMIT 1;

        EXIT WHEN best_rank IS NULL;

        -- Merge all non-overlapping occurrences of that pair.
        nw := ARRAY[]::TEXT[];
        i := 1;
        n := array_length(w, 1);
        WHILE i <= n LOOP
            IF i < n AND w[i] = best_left AND w[i + 1] = best_right THEN
                nw := nw || (best_left || best_right);
                i := i + 2;
            ELSE
                nw := nw || w[i];
                i := i + 1;
            END IF;
        END LOOP;

        w := nw;
        EXIT WHEN array_length(w, 1) = 1;
    END LOOP;

    RETURN w;
END;
$func$ LANGUAGE plpgsql STABLE;

-- Encode free text into an array of GPT-2 token ids.
-- NOTE: GPT-2's pre-tokenization regex uses \p{L}/\p{N} and a negative
-- lookahead that Postgres regex can't express. We approximate with POSIX
-- classes ([[:alpha:]]/[[:digit:]], which honor the UTF-8 locale) and a plain
-- whitespace run. Identical to GPT-2 for normal single-spaced text; unusual
-- whitespace runs may tokenize slightly differently.
CREATE OR REPLACE FUNCTION gpt2_encode(p_text TEXT)
RETURNS INT[] AS $func$
DECLARE
    ids INT[] := ARRAY[]::INT[];
    chunk TEXT;
    tok TEXT;
BEGIN
    FOR chunk IN
        SELECT (m.match)[1]
        FROM regexp_matches(
            p_text,
            $re$'s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^[:space:][:alpha:][:digit:]]+|[[:space:]]+$re$,
            'g'
        ) WITH ORDINALITY AS m(match, ord)
        ORDER BY m.ord
    LOOP
        FOREACH tok IN ARRAY gpt2_bpe(gpt2_byte_symbols(chunk)) LOOP
            ids := ids || (SELECT token_id FROM bpe_vocab WHERE token = tok);
        END LOOP;
    END LOOP;

    RETURN ids;
END;
$func$ LANGUAGE plpgsql STABLE;
