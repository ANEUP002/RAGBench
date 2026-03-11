import csv
from pathlib import Path


def save_results(results, file_path="results/benchmark_results.csv"):

    results_path = Path(file_path)
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "index_type",
                "chunk_size",
                "top_k",
                "precision",
                "recall",
                "hit_rate",
                "mrr",
                "f1",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark results saved to: {results_path}")