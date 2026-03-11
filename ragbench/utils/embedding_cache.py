import os
import numpy as np


def get_cache_path(model_name, chunk_size):

    model_safe = model_name.replace("/", "_")

    return f"embeddings_cache/{model_safe}_chunks_{chunk_size}.npy"


def load_or_compute_embeddings(embedder, texts, model_name, chunk_size):

    os.makedirs("embeddings_cache", exist_ok=True)

    cache_path = get_cache_path(model_name, chunk_size)

    if os.path.exists(cache_path):

        print(f"Loading cached embeddings: {cache_path}")

        return np.load(cache_path)

    print("Computing embeddings...")

    embeddings = embedder.encode(texts)

    np.save(cache_path, embeddings)

    print(f"Saved embeddings → {cache_path}")

    return embeddings