# RAGBench
Benchmarking Retrieval Pipelines for Retrieval-Augmented Generation

RAGBench is a modular benchmarking framework for evaluating dense retrieval pipelines used in Retrieval-Augmented Generation (RAG) systems.

It supports systematic experimentation across embedding models, vector search indexes, and chunking strategies while measuring retrieval quality using standard Information Retrieval metrics.

Built for researchers and engineers exploring high-performance RAG architectures.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FAISS](https://img.shields.io/badge/vector_search-FAISS-orange)
![License](https://img.shields.io/badge/license-MIT-green)
## Benchmark Results

### Chunk Size vs Retrieval Quality

![Chunk Size vs F1](results/plots/chunk_vs_f1.png)

### Index Type vs Recall

![Index vs Recall](results/plots/index_vs_recall.png)

### Model Comparison

![Model vs F1](results/plots/model_vs_f1.png)
