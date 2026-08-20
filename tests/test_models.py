import numpy as np

from failure_risk.models import WeibullAFTModel


def test_weibull_aft_survival_monotonic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(250, 3))
    duration = np.exp(3.0 + 0.2 * X[:, 0] + rng.normal(scale=0.35, size=250))
    censor = rng.uniform(10, 40, size=250)
    event = (duration <= censor).astype(int)
    observed = np.minimum(duration, censor)

    model = WeibullAFTModel().fit(X, observed, event)
    assert model.converged_
    row = X[:1]
    s12 = model.predict_survival(row, np.array([12.0]))[0]
    s24 = model.predict_survival(row, np.array([24.0]))[0]
    assert 0 <= s24 <= s12 <= 1
