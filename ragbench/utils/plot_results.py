import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots(csv_path="results/benchmark_results.csv"):

    df = pd.read_csv(csv_path)

    os.makedirs("results/plots", exist_ok=True)

    # 1️⃣ Chunk size vs F1
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x="chunk_size",
        y="f1",
        hue="model",
        style="index_type",
        markers=True
    )

    plt.title("Chunk Size vs F1 Score")
    plt.savefig("results/plots/chunk_vs_f1.png")
    plt.close()


    # 2️⃣ Index type comparison
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=df,
        x="index_type",
        y="recall",
        hue="model"
    )

    plt.title("Index Type vs Recall")
    plt.savefig("results/plots/index_vs_recall.png")
    plt.close()


    # 3️⃣ Model comparison
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=df,
        x="model",
        y="f1",
        hue="index_type"
    )

    plt.title("Model vs F1 Score")
    plt.xticks(rotation=30)
    plt.savefig("results/plots/model_vs_f1.png")
    plt.close()

    print("Plots saved in results/plots/")