# src/evaluation.py
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy:", acc)
    print("F1 score:", f1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    return {"accuracy": acc, "f1": f1}


def plot_confusion(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.show()