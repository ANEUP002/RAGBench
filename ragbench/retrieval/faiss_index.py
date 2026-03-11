import faiss
import numpy as np


class FAISSIndex:
    def __init__(self, index_type="flat", dim=None, nlist=2, m=8, hnsw_m=32):
        self.index_type = index_type.lower()
        self.dim = dim
        self.nlist = nlist  ##number of clusters for IVF so that similar embeddings are arranged into same clusters, good for scalibility
        self.m = m ##number of subvector partitions
        self.hnsw_m = hnsw_m ##number of graph vectors
        self.index = None

    def build(self, embeddings):
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype="float32")

        embeddings = embeddings.astype("float32")
        self.dim = embeddings.shape[1]

        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dim)

        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
            self.index.train(embeddings)

        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)

        elif self.index_type == "pq":
            self.index = faiss.IndexPQ(self.dim, self.m, 8)
            self.index.train(embeddings)

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        self.index.add(embeddings)

    def search(self, query_embeddings, top_k=5):
        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings, dtype="float32")

        query_embeddings = query_embeddings.astype("float32")

        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices