from ragbench.experiments.experiment_runner import ExperimentRunner


def main():

    runner = ExperimentRunner(
        dataset_path="datasets/scifact",   # dataset folder
        models_path="config/models.yaml",
        index_types=["flat", "hnsw", "ivf"],
        chunk_sizes=[128, 256, 512],
        top_k_values=[3],
    )

    results = runner.run()

    print("\nAll Experiment Results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()