from transformers import GPT2LMHeadModel, AutoTokenizer
import csv
import json

print("Loading GPT-2 Small for pgvector export...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
state_dict = model.state_dict()
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# this is neither fast nor memory efficient, but see goals
def to_pg_array(ndarray):
    # Formats a numpy array into a postgres vector string: '[1.0,2.0,3.0...]'
    return "[" + ",".join(map(str, ndarray.tolist())) + "]"


def bytes_to_unicode():
    # GPT-2's canonical byte -> unicode map. Reconstructed here so the export
    # doesn't depend on tokenizer internals that move between library versions.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# 1. Export Token Embeddings
print("Exporting Token Embeddings...")
with open("token_embeddings_vector.csv", "w", newline="") as f:
    writer = csv.writer(f)
    wte = state_dict["transformer.wte.weight"].numpy()
    for token_id in range(wte.shape[0]):
        writer.writerow([token_id, to_pg_array(wte[token_id])])

# 1b. Export Position Embeddings (wpe): one 768-dim vector per position 0..1023
print("Exporting Position Embeddings...")
with open("position_embeddings_vector.csv", "w", newline="") as f:
    writer = csv.writer(f)
    wpe = state_dict["transformer.wpe.weight"].numpy()
    for position in range(wpe.shape[0]):
        writer.writerow([position, to_pg_array(wpe[position])])

# 2. Export Layer Weights with 768-chunking
print("Exporting Layer Weights...")
with open("layer_weights_vector.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for key, tensor in state_dict.items():
        if not key.startswith("transformer.h."):
            continue

        parts = key.split(".")
        layer_idx = int(parts[2])
        tensor_name = ".".join(parts[3:])
        t = tensor.numpy()

        # HuggingFace weights are often transposed (Shape: [In, Out])
        # We want to iterate through the target output dimensions
        if len(t.shape) == 2:
            if tensor_name == "attn.c_attn.weight":
                # Fused QKV projection, shape [768, 2304], where the 2304
                # outputs are Q|K|V stacked (768 each). Encode which block via
                # chunk_idx (0=Q, 1=K, 2=V) with row_idx 0..767 inside the
                # block, storing each output's 768-dim input-weight column.
                # This is exactly the layout attention.sql's qkv_proj expects.
                for qkv_type in range(3):
                    for row_idx in range(768):
                        col = t[:, qkv_type * 768 + row_idx]
                        writer.writerow(
                            [layer_idx, tensor_name, row_idx, qkv_type, to_pg_array(col)]
                        )
            else:
                # Every other 2D weight is stored COLUMN-MAJOR: one 768-dim
                # input-weight column per output neuron, so a forward pass is a
                # dot product (p_x <#> vec). row_idx = output index; when the
                # input dim > 768 it is split into 768-wide pieces via chunk_idx.
                #   mlp.c_fc.weight   [768, 3072] -> row_idx 0..3071, chunk_idx 0
                #   mlp.c_proj.weight [3072, 768] -> row_idx 0..767,  chunk_idx 0..3
                #   attn.c_proj.weight [768, 768] -> row_idx 0..767,  chunk_idx 0
                # GPT-2's dims are all multiples of 768, so this divides evenly.
                n_chunks = t.shape[0] // 768
                for row_idx in range(t.shape[1]):
                    col = t[:, row_idx]
                    for chunk_idx in range(n_chunks):
                        chunk_slice = col[chunk_idx * 768 : (chunk_idx + 1) * 768]
                        writer.writerow(
                            [
                                layer_idx,
                                tensor_name,
                                row_idx,
                                chunk_idx,
                                to_pg_array(chunk_slice),
                            ]
                        )

        elif len(t.shape) == 1:
            # 1D tensors: ln_1/ln_2 weight & bias (768) plus all biases
            # (attn.c_attn.bias is 2304, mlp.c_fc.bias is 3072, others 768).
            # Stored at row_idx=0, chunked to 768 so layernorm() can read
            # ln_*.weight / ln_*.bias at (row_idx=0, chunk=0).
            n_chunks = t.shape[0] // 768
            for chunk_idx in range(n_chunks):
                chunk_slice = t[chunk_idx * 768 : (chunk_idx + 1) * 768]
                writer.writerow(
                    [layer_idx, tensor_name, 0, chunk_idx, to_pg_array(chunk_slice)]
                )

    # transformer.ln_f is the final LayerNorm applied after all 12 blocks (it
    # lives outside transformer.h.*, so the loop above skips it). generate.sql
    # reads it as layernorm(11, 'ln_f', ...), so store it under layer_idx=11
    # with tensor_name ln_f.weight / ln_f.bias (both 768-dim).
    for suffix in ("weight", "bias"):
        ln_f = state_dict[f"transformer.ln_f.{suffix}"].numpy()
        writer.writerow([11, f"ln_f.{suffix}", 0, 0, to_pg_array(ln_f)])

# 3. Export the vocabulary (token_id -> decoded text) so generate.sql can turn
# generated token ids back into readable text. Byte-level tokens that aren't
# valid UTF-8 on their own decode to replacement/control chars; fine for a v1
# demo. QUOTE_ALL keeps empty/control/comma fields COPY-safe, and we strip NUL
# bytes (\x00) since Postgres COPY refuses to store them.
print("Exporting Vocab...")
with open("vocab_vector.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for token_id in range(len(tokenizer)):
        piece = tokenizer.decode([token_id]).replace("\x00", "")
        writer.writerow([token_id, piece])

# 4. Export the BPE encoder assets so gpt2_encode() can tokenize text in SQL:
#   - byte_encoder: 256-entry byte -> unicode-char map (bytes_to_unicode)
#   - bpe_vocab:    byte-char token string -> token id (get_vocab())
#   - bpe_merges:   ranked merge pairs from the tokenizer model; rank = order
# Both vocab and merges are pulled from the fast tokenizer's serialized backend
# model, which is stable across library versions. These byte-char strings are
# all printable (no NUL) but can contain ',' and '"', so QUOTE_ALL keeps them
# COPY-safe.
print("Exporting BPE encoder assets...")

with open("byte_encoder.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for byte, uchar in bytes_to_unicode().items():
        writer.writerow([byte, uchar])

with open("bpe_vocab.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for token, token_id in tokenizer.get_vocab().items():
        writer.writerow([token, token_id])

# The serialized backend model contains the rank-ordered merge list. Depending
# on the tokenizers version each merge is either a "left right" string or a
# [left, right] pair; handle both. Byte-char symbols never contain a space.
raw_merges = json.loads(tokenizer.backend_tokenizer.to_str())["model"]["merges"]

with open("bpe_merges.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    rank = 0
    for m in raw_merges:
        parts = m.split(" ") if isinstance(m, str) else list(m)
        if len(parts) == 2:
            writer.writerow([rank, parts[0], parts[1]])
            rank += 1

print("Vector CSV generation complete.")
