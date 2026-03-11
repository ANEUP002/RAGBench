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


---

# Supported Embedding Models

Current models tested:

- sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-base-en-v1.5
- hkunlp/instructor-large

These models generate vector embeddings used for similarity search.

---

# Vector Index Types

FAISS indexes supported:

| Index | Description |
|------|-------------|
| Flat | Exact nearest neighbor search |
| HNSW | Graph-based approximate nearest neighbors |
| IVF | Inverted file index for large datasets |

---

# Evaluation Metrics

RAGBench evaluates retrieval performance using:

- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- F1 Score

---

# Dataset

Experiments are performed on the **SciFact dataset** from the BEIR benchmark.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/yourusername/RAGBench.git
cd RAGBench

Create Environment

-python -m venv venv

Activate

-venv\Scripts\activate

Install dependencies:

-pip install -r requirements.txt

Run benchmark:

python main.py


This removes friction for anyone testing your code.

---

# 4️⃣ Add an Architecture Diagram (very impressive)

Even a simple diagram helps.

Add:

```markdown
## System Architecture

Dataset
↓
Document Chunking
↓
Embedding Model
↓
Vector Index (FAISS)
↓
Top-K Retrieval
↓
Evaluation Metrics
↓
Benchmark Results + Visualization



