import numpy as np

from metrics import bootstrap_intervals, grouped_bootstrap, summarize_metrics

CLASSES = ["a", "b", "c"]


def one_hot(pred, n=3):
    probs = np.full((len(pred), n), 0.05)
    probs[np.arange(len(pred)), pred] = 0.9
    return probs


def test_perfect_predictions_give_a_point_interval():
    y = np.array([0, 1, 2] * 20)
    ci = bootstrap_intervals(y, one_hot(y), CLASSES, n_boot=200)
    assert ci["accuracy"] == [1.0, 1.0] and ci["balanced_accuracy"] == [1.0, 1.0]
    assert ci["sensitivity"]["b"] == [1.0, 1.0]


def test_interval_brackets_the_estimate_and_narrows_with_more_data():
    rng = np.random.default_rng(1)
    def run(n):
        y = rng.integers(0, 3, n)
        pred = np.where(rng.random(n) < 0.7, y, (y + 1) % 3)
        point = summarize_metrics(y, one_hot(pred), CLASSES)
        low, high = point["confidence_intervals_95"]["accuracy"]
        assert low <= point["accuracy"] <= high
        return high - low
    assert run(2000) < run(100)


def test_groups_are_resampled_whole():
    # Ten groups of 30 identical rows: a grouped bootstrap must see the
    # group-to-group spread, which is much wider than an image-level one.
    y = np.repeat(np.arange(10) % 3, 30)
    correct = np.repeat(np.arange(10) < 7, 30)
    pred = np.where(correct, y, (y + 1) % 3)
    groups = np.repeat(np.arange(10), 30)
    by_image = bootstrap_intervals(y, one_hot(pred), CLASSES, n_boot=300)["accuracy"]
    by_group = bootstrap_intervals(y, one_hot(pred), CLASSES, groups=groups, n_boot=300)["accuracy"]
    assert (by_group[1] - by_group[0]) > 2 * (by_image[1] - by_image[0])


def test_grouped_bootstrap_drops_metrics_that_are_usually_undefined():
    groups = np.arange(100)
    rare = np.zeros(100, dtype=bool)
    rare[0] = True
    out = grouped_bootstrap(lambda rows: {"always": 1.0, "rare": 1.0 if rare[rows].any() else None}, groups, n_boot=200)
    assert out["always"] == [1.0, 1.0]
    assert out["rare"] is not None  # present in ~63% of resamples, above the 50% bar
