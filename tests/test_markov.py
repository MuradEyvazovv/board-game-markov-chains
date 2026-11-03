import numpy as np
import pytest

from boardchains import markov


def two_state(p=0.3, q=0.1):
    return np.array([[1 - p, p], [q, 1 - q]])


def test_stationary_two_state_closed_form():
    p, q = 0.3, 0.1
    P = two_state(p, q)
    expected = np.array([q, p]) / (p + q)
    for pi in (markov.stationary_linear(P), markov.stationary_power(P)[0], markov.stationary_eigen(P)):
        assert np.allclose(pi, expected, atol=1e-12)


def test_assert_stochastic_rejects_bad_rows():
    P = two_state()
    P[0, 0] += 0.01
    with pytest.raises(ValueError):
        markov.assert_stochastic(P)


def test_absorbing_geometric():
    # Leave state 0 with probability 1/4 each step: steps ~ Geometric(1/4).
    a = 0.25
    P = np.array([[1 - a, a], [0.0, 1.0]])
    res = markov.absorbing_analysis(P, np.array([0]))
    assert res["expected_steps"][0] == pytest.approx(1 / a)
    assert res["variance"][0] == pytest.approx((1 - a) / a**2)
    cdf = markov.absorption_cdf(P, start=0, absorbing=1)
    t = np.arange(len(cdf))
    assert np.allclose(cdf, 1 - (1 - a) ** t)
