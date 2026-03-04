# RAGBench
# RAGBench

RAGBench is a lightweight benchmarking framework for evaluating Retrieval-Augmented Generation (RAG) systems.

The goal of this project is to measure retrieval performance using standard information retrieval metrics such as:

* Precision
* Recall
* F1 Score

RAGBench simulates a retrieval pipeline and evaluates how well a system retrieves relevant documents for a given query.

## Features

* Evaluate retrieval quality
* Compute precision, recall, and F1
* Modular retrieval pipeline
* Easy integration with vector databases (FAISS, Chroma, etc.)

## Project Structure

```
RAGBench/
│
├── dataset.py        # Example evaluation dataset
├── retrieval.py      # Retrieval simulation / retrieval system
├── metrics.py        # Precision, recall, F1 implementation
├── evaluate.py       # Runs evaluation pipeline
└── README.md
```

## Future Improvements

* Embedding-based retrieval
* FAISS vector search
* Top-K evaluation
* Hit@K, MRR, nDCG metrics
* Visualization of benchmark results

## Goal

The goal of RAGBench is to provide a simple framework for experimenting with and evaluating RAG systems.

## Author

Ayush Neupane
