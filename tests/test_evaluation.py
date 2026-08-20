import numpy as np

from failure_risk.evaluation import concordance_index


def test_concordance_perfect_ordering():
    duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    assert concordance_index(duration, event, risk) == 1.0
