import numpy as np
import pandas as pd

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

        # convertir probabilidades en clases
        y_pred = (y_prob >= threshold).astype(int)

        # metricas
        acc = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        # guardar resultados
        results.append({
            "threshold": round(threshold, 2),
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        })

    return pd.DataFrame(results)