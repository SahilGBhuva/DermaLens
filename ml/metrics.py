from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, probabilities, bins=10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correctness = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        if not mask.any():
            continue

        bin_accuracy = correctness[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += mask.mean() * abs(bin_accuracy - bin_confidence)

    return float(ece)


def multiclass_brier_score(y_true, probabilities, num_classes):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    one_hot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def per_class_sensitivity_specificity(confusion, class_names):
    confusion = np.asarray(confusion)
    total = confusion.sum()
    output = {}

    for i, class_name in enumerate(class_names):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        tn = total - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        output[class_name] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }

    return output


def summarize_metrics(y_true, probabilities, class_names):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    y_pred = probabilities.argmax(axis=1)
    confusion = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "multiclass_brier_score": multiclass_brier_score(
            y_true, probabilities, len(class_names)
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class_sensitivity_specificity": per_class_sensitivity_specificity(
            confusion, class_names
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    try:
        metrics["macro_ovr_roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=list(range(len(class_names))),
            )
        )
    except ValueError:
        metrics["macro_ovr_roc_auc"] = None

    return metrics
