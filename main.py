from ragbench.datasets.dataset_loader import DataLoader
from ragbench.utils.chunking import chunk_documents
def main():
    loader = DataLoader(
        "data/corpus.json",
        "data/queries.json",
        "data/ground_truth.json"
    )

    corpus, queries, ground_truth = loader.load_dataset()

    chunks = chunk_documents(corpus)
    print("Chunks:")
    for c in chunks:
        print(c)

    
if __name__ == "__main__":
    main()