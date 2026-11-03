import numpy as np
import pytest

from boardchains import markov, snakes


@pytest.mark.parametrize("rule", snakes.RULES)
def test_rows_are_stochastic(rule):
    P = snakes.transition_matrix(rule)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-14)
    assert P[snakes.FINISH, snakes.FINISH] == 1.0


def test_jump_squares_are_never_occupied():
    P = snakes.transition_matrix("stay")
    for square in snakes.JUMPS:
        assert P[:, square].sum() == 0.0


@pytest.mark.parametrize("rule", snakes.RULES)
def test_expected_turns_is_fundamental_matrix_row_sum(rule):
    P = snakes.transition_matrix(rule)
    tr = snakes.transient_states()
    N = np.linalg.inv(np.eye(len(tr)) - P[np.ix_(tr, tr)])
    start = int(np.where(tr == 0)[0][0])
    g = snakes.analyze(rule)
    assert g.expected_turns == pytest.approx(N[start].sum(), rel=1e-12)
    assert g.expected_from_pmf == pytest.approx(g.expected_turns, rel=1e-9)
    assert g.variance_from_pmf == pytest.approx(g.variance, rel=1e-8)


def test_distribution_is_a_distribution():
    g = snakes.analyze("stay")
    assert (np.diff(g.cdf) >= -1e-15).all()
    assert g.cdf[-1] == pytest.approx(1.0, abs=1e-12)
    assert g.cdf[g.median] >= 0.5 > g.cdf[g.median - 1]
    assert g.shortest_game == 7


def test_finishing_rules_order():
    free = snakes.analyze("free_finish").expected_turns
    stay = snakes.analyze("stay").expected_turns
    bounce = snakes.analyze("bounce").expected_turns
    assert free < stay < bounce


def test_monte_carlo_mean_within_ci():
    g = snakes.analyze("stay")
    lengths = snakes.simulate(200_000, seed=5, rule="stay")
    se = lengths.std(ddof=1) / np.sqrt(len(lengths))
    assert abs(lengths.mean() - g.expected_turns) < 4 * se
    assert lengths.min() >= g.shortest_game


def test_absorption_cdf_matches_simulation_shape():
    g = snakes.analyze("stay")
    lengths = snakes.simulate(200_000, seed=6, rule="stay")
    ecdf = np.searchsorted(np.sort(lengths), np.arange(len(g.cdf)), side="right") / len(lengths)
    assert np.abs(ecdf - g.cdf).max() < 0.01


def test_two_player_first_mover_edge():
    g = snakes.analyze("stay")
    assert 0.5 < g.two_player_first_wins < 0.52
    assert g.two_player_expected_rounds < g.expected_turns


def test_chute_48_variant():
    # The only difference is where the 26 chute starts.
    assert set(snakes.JUMPS) ^ set(snakes.JUMPS_48) == {47, 48}
    assert snakes.JUMPS_48[48] == 26
    g = snakes.analyze("stay", snakes.JUMPS_48)
    assert g.expected_turns == pytest.approx(39.598365640208, rel=1e-10)
    assert g.expected_from_pmf == pytest.approx(g.expected_turns, rel=1e-9)
