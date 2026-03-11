from ragbench.datasets.dataset_loader import DataLoader ##For testing
from ragbench.datasets.beir_loader import BEIRLoader ##For actual implementation
from ragbench.utils.chunking import chunk_documents
from ragbench.utils.config_loader import load_models
from ragbench.embeddings.embedder import Embedder
from ragbench.retrieval.faiss_index import FAISSIndex
from ragbench.retrieval.retriever import Retriever
from ragbench.evaluation.evaluator import Evaluator
from results.benchmark_results import save_results
from ragbench.utils.embedding_cache import load_or_compute_embeddings


class Benchmark:

    def __init__(self, dataset_path, models_path, top_k=3, index_type="flat", chunk_size=256):

        self.dataset_path = dataset_path
        self.models_path = models_path
        self.top_k = top_k
        self.index_type = index_type
        self.chunk_size = chunk_size

    def run(self):

        loader = BEIRLoader(self.dataset_path)

        corpus, queries, ground_truth = loader.load()

        chunks = chunk_documents(corpus, chunk_size=self.chunk_size)

        print(f"Total chunks created: {len(chunks)}")

        texts = [chunk["text"] for chunk in chunks]

        models = load_models(self.models_path)

        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        all_results = []

        for model_name in models:

            print(f"\nRunning benchmark for: {model_name}")

            embedder = Embedder(model_name)

            doc_embeddings = load_or_compute_embeddings(
                embedder,
                texts,
                model_name,
                self.chunk_size,
            )

            index = FAISSIndex(index_type=self.index_type)
            index.build(doc_embeddings)

            query_embeddings = embedder.encode(query_texts)

            distances, indices = index.search(query_embeddings, top_k=self.top_k)

            retriever = Retriever(chunks)
            retrieved_docs = retriever.map_indices_to_doc_ids(indices)

            evaluator = Evaluator(ground_truth, k=self.top_k)

            metrics = evaluator.evaluate(query_ids, retrieved_docs)

            result = {
                "model": model_name,
                "index_type": self.index_type,
                "chunk_size": self.chunk_size,
                "top_k": self.top_k,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "hit_rate": metrics["hit_rate"],
                "mrr": metrics["mrr"],
                "f1": metrics["f1"],
            }

            all_results.append(result)

        return all_results

