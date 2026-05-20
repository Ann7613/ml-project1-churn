import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_thresholds(y_true, y_prob):

    results = []
    thresholds = np.arange(0.1, 0.95, 0.05)

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": round(threshold, 2),
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        })

    return pd.DataFrame(results)


def get_best_threshold(threshold_results, metric="f1"):

    best_row = threshold_results.sort_values(
        by=metric,
        ascending=False
    ).iloc[0]

    best_threshold = best_row["threshold"]

    return best_threshold, best_row


def plot_threshold_metrics(threshold_results, model_name="Modelo"):

    plt.figure(figsize=(7, 5))

    plt.plot(
        threshold_results["threshold"],
        threshold_results["f1"],
        marker="o",
        label="F1-score"
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        marker="o",
        label="Precision"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Métricas según Threshold - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)

    filename = model_name.replace(" ", "_")
    plt.savefig(f"figures/threshold_metrics_{filename}.png", dpi=300)

    plt.show()