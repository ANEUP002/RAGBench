from ragbench.pipeline.benchmark import Benchmark
from results.benchmark_results import save_results
from ragbench.utils.plot_results import generate_plots


class ExperimentRunner:

    def __init__(
        self,
        dataset_path,
        models_path,
        index_types,
        chunk_sizes,
        top_k_values,
    ):
        self.dataset_path = dataset_path
        self.models_path = models_path
        self.index_types = index_types
        self.chunk_sizes = chunk_sizes
        self.top_k_values = top_k_values

    def run(self):

        all_results = []

        for chunk_size in self.chunk_sizes:
            for index_type in self.index_types:
                for top_k in self.top_k_values:

                    print(
                        f"\nRunning experiment: chunk_size={chunk_size}, index={index_type}, top_k={top_k}"
                    )

                    benchmark = Benchmark(
                        dataset_path=self.dataset_path,
                        models_path=self.models_path,
                        top_k=top_k,
                        index_type=index_type,
                        chunk_size=chunk_size,
                    )

                    results = benchmark.run()

                    all_results.extend(results)

        save_results(all_results)
        generate_plots()

        return all_results