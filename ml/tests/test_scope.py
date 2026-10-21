import numpy as np

from fit_scope import fit_gaussian, scope_scores


def test_points_near_a_class_score_low_and_far_points_score_high():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    labels = np.repeat([0, 1], 300)
    features = centres[labels] + rng.normal(0, 1, (600, 3))
    means, precision = fit_gaussian(features, labels, n_classes=3)  # class 2 never seen
    assert not np.isfinite(means[2]).all()

    near = scope_scores(np.array([[0.2, -0.1, 0.0], [9.8, 0.3, 0.1]]), means, precision)
    far = scope_scores(np.array([[5.0, 30.0, -30.0]]), means, precision)
    assert near.max() < 3 and far[0] > 100


def test_threshold_from_validation_quantile_accepts_the_target_share():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 2000)
    features = np.stack([labels * 6.0, np.zeros(2000)], axis=1) + rng.normal(0, 1, (2000, 2))
    means, precision = fit_gaussian(features[:1000], labels[:1000], n_classes=2)
    val = scope_scores(features[1000:], means, precision)
    threshold = np.quantile(val, 0.95)
    assert abs((val <= threshold).mean() - 0.95) < 0.01
