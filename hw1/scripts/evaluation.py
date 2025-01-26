import numpy as np


def classification_accuracy(predictions: np.array, true_labels: np.array) -> float:
    assert (
        predictions.shape[0] == true_labels.shape[0]
    ), "Predictions and true labels are of different lengths"
    return np.count_nonzero(predictions == true_labels) / predictions.shape[0]
