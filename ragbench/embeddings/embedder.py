import numpy as np
from ragbench.embeddings.model_loader import load_model


class Embedder:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = load_model(model_name)

    def encode(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings.astype("float32")