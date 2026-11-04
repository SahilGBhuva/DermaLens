import numpy as np

from selective import CLASSES, balanced_accuracy, selective_curve


def make(n=400, seed=0):
    """Confident predictions are right more often than unsure ones."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, len(CLASSES), n)
    conf = rng.uniform(0.3, 1.0, n)
    right = rng.uniform(size=n) < conf
    pred = np.where(right, truth, (truth + 1) % len(CLASSES))
    probs = np.full((n, len(CLASSES)), 0.0)
    for i in range(n):
        probs[i, :] = (1 - conf[i]) / (len(CLASSES) - 1)
        probs[i, pred[i]] = conf[i]
    return probs, truth, np.arange(n)


def test_full_coverage_answers_everything_and_accuracy_rises_as_coverage_falls():
    probs, truth, groups = make()
    rows = selective_curve(probs, truth, groups, n_boot=50)
    assert rows[0]["coverage"] == 1.0 and rows[0]["answered"] == len(truth)
    accuracies = [r["accuracy"] for r in rows]
    assert accuracies[-1] > accuracies[0]


def test_every_melanoma_is_accounted_for():
    probs, truth, groups = make()
    for row in selective_curve(probs, truth, groups, n_boot=20):
        total = row["melanomas_answered_correctly"] + row["melanomas_answered_wrongly"] + row["melanomas_referred"]
        assert total == row["melanomas"] == int((truth == CLASSES.index("mel")).sum())


def test_referred_melanomas_grow_as_coverage_shrinks():
    probs, truth, groups = make()
    rows = selective_curve(probs, truth, groups, n_boot=20)
    assert rows[0]["melanomas_referred"] == 0
    assert rows[-1]["melanomas_referred"] >= rows[0]["melanomas_referred"]


def test_balanced_accuracy_ignores_absent_classes():
    truth = np.array([0, 0, 1])
    pred = np.array([0, 1, 1])
    assert balanced_accuracy(truth, pred) == 0.75
