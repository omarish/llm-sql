from transformers import GPT2LMHeadModel
import csv

print("Loading GPT-2 Small for pgvector export...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
state_dict = model.state_dict()


# this is neither fast nor memory efficient, but see goals
def to_pg_array(ndarray):
    # Formats a numpy array into a postgres vector string: '[1.0,2.0,3.0...]'
    return "[" + ",".join(map(str, ndarray.tolist())) + "]"


# 1. Export Token Embeddings
print("Exporting Token Embeddings...")
with open("token_embeddings_vector.csv", "w", newline="") as f:
    writer = csv.writer(f)
    wte = state_dict["transformer.wte.weight"].numpy()
    for token_id in range(wte.shape[0]):
        writer.writerow([token_id, to_pg_array(wte[token_id])])

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
            # If it's an MLP layer scaling to 3072, chunk it into 4x768 blocks
            if t.shape[1] == 3072:
                for row_idx in range(t.shape[0]):
                    for chunk in range(4):
                        chunk_slice = t[row_idx, chunk * 768 : (chunk + 1) * 768]
                        writer.writerow(
                            [
                                layer_idx,
                                tensor_name,
                                row_idx,
                                chunk,
                                to_pg_array(chunk_slice),
                            ]
                        )
            else:
                # Normal 768x768 layer
                for row_idx in range(t.shape[1]):
                    writer.writerow(
                        [layer_idx, tensor_name, row_idx, 0, to_pg_array(t[:, row_idx])]
                    )

print("Vector CSV generation complete.")
