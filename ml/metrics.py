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


def grouped_bootstrap(statistic, groups, n_boot=1000, seed=0, level=0.95):
    """Percentile intervals for `statistic(rows) -> {name: value or None}`,
    resampling whole groups (patients, lesions) with replacement."""
    _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(group_index == g) for g in range(group_index.max() + 1)]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        for name, value in statistic(rows).items():
            if value is not None:
                draws.setdefault(name, []).append(float(value))
    tail = (1 - level) / 2 * 100
    return {
        name: [round(float(v), 4) for v in np.percentile(values, [tail, 100 - tail])]
        if len(values) >= n_boot * 0.5 else None
        for name, values in draws.items()
    }


def bootstrap_intervals(y_true, probabilities, class_names, groups=None, n_boot=1000, seed=0, level=0.95):
    """Percentile bootstrap intervals for the headline metrics.

    With `groups` (lesion or patient ids) whole groups are resampled, because
    several images of one lesion are not independent evidence. Classes absent
    from a resample are skipped for that resample's per-class values.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    y_pred = probabilities.argmax(axis=1)
    n_classes = len(class_names)
    if groups is None:
        groups = np.arange(len(y_true))
    _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(group_index == g) for g in range(group_index.max() + 1)]

    rng = np.random.default_rng(seed)
    draws = {"accuracy": [], "balanced_accuracy": [], "expected_calibration_error": []}
    per_class = {name: [] for name in class_names}
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), len(members))
        rows = np.concatenate([members[g] for g in picked])
        yt, yp = y_true[rows], y_pred[rows]
        recalls = []
        for k, name in enumerate(class_names):
            mask = yt == k
            if mask.any():
                recall = float((yp[mask] == k).mean())
                recalls.append(recall)
                per_class[name].append(recall)
        draws["accuracy"].append(float((yt == yp).mean()))
        draws["balanced_accuracy"].append(float(np.mean(recalls)))
        draws["expected_calibration_error"].append(expected_calibration_error(yt, probabilities[rows]))

    tail = (1 - level) / 2 * 100

    def interval(values):
        if len(values) < n_boot * 0.5:  # class too rare to resample reliably
            return None
        low, high = np.percentile(values, [tail, 100 - tail])
        return [round(float(low), 4), round(float(high), 4)]

    return {
        "level": level,
        "resamples": n_boot,
        "unit": "group" if len(members) < len(y_true) else "image",
        **{name: interval(values) for name, values in draws.items()},
        "sensitivity": {name: interval(values) for name, values in per_class.items()},
    }


def summarize_metrics(y_true, probabilities, class_names, groups=None):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    y_pred = probabilities.argmax(axis=1)
    # Pin every class so the matrix stays 7x7 even when a rare class is absent
    # from the split; otherwise per-class rows would shift out of alignment.
    labels = list(range(len(class_names)))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
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
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    # ROC-AUC is undefined when a class is missing from the split. Newer
    # scikit-learn returns NaN instead of raising; store None either way,
    # because NaN is not valid JSON and the API cannot serve it.
    try:
        auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        )
        metrics["macro_ovr_roc_auc"] = auc if np.isfinite(auc) else None
    except ValueError:
        metrics["macro_ovr_roc_auc"] = None

    metrics["confidence_intervals_95"] = bootstrap_intervals(y_true, probabilities, class_names, groups)
    return metrics
