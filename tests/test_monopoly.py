import numpy as np
import pytest
from scipy import stats

from boardchains.monopoly import board, chain, economics, published, simulate


@pytest.fixture(scope="module", params=chain.POLICIES)
def solved(request):
    return chain.solve(request.param)


def test_dice_outcomes_sum_to_one():
    outcomes = chain.dice_outcomes()
    assert sum(p for _, _, p in outcomes) == pytest.approx(1.0)
    assert sum(p for _, dbl, p in outcomes if dbl) == pytest.approx(1 / 6)


@pytest.mark.parametrize("policy", chain.POLICIES)
def test_rows_are_stochastic(policy):
    P = chain.transition_matrix(policy)
    assert P.shape == (chain.N_STATES, chain.N_STATES)
    assert (P >= 0).all()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-14)


def test_stationary_sums_to_one_and_is_fixed_point(solved):
    pi = solved.pi_states
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert (pi >= 0).all()
    assert np.allclose(pi @ solved.P, pi, atol=1e-14)
    assert solved.pi.sum() == pytest.approx(1.0, abs=1e-12)


def test_three_solvers_agree(solved):
    assert solved.max_diff_power < 1e-12
    assert solved.max_diff_eigen < 1e-12


def test_nobody_rests_on_go_to_jail(solved):
    assert solved.pi[board.GO_TO_JAIL] == 0.0


def test_chance_back_three_chains_into_community_chest():
    out = chain.resolve_landing(36)
    # Go: the Chance card, or back 3 to Community Chest 3 and then its Go card.
    assert out[board.GO] == pytest.approx(1 / 16 + 1 / 16 * 1 / 16)
    assert out[board.IN_JAIL] == pytest.approx(1 / 16 + 1 / 16 * 1 / 16)
    assert out[33] == pytest.approx(1 / 16 * 14 / 16)
    assert out[36] == pytest.approx(6 / 16)
    assert out[5] == pytest.approx(3 / 16)  # Reading is the nearest railroad from 36, plus its own card
    assert sum(out.values()) == pytest.approx(1.0)


def test_back_three_elsewhere_is_plain():
    assert chain.resolve_landing(7)[4] == pytest.approx(1 / 16)  # Income Tax
    assert chain.resolve_landing(22)[19] == pytest.approx(1 / 16)  # New York Avenue


def test_leave_early_never_uses_later_jail_turns():
    res = chain.solve("leave_early")
    assert res.pi_states[chain.J2] == 0.0
    assert res.pi_states[chain.J3] == 0.0


def test_staying_in_jail_means_more_time_in_jail():
    stay = chain.solve("stay").pi[board.IN_JAIL]
    early = chain.solve("leave_early").pi[board.IN_JAIL]
    assert stay > 2 * early


@pytest.mark.parametrize("policy", chain.POLICIES)
def test_monte_carlo_matches_exact(policy):
    exact = chain.solve(policy).pi
    sim = simulate.simulate(policy, n_chains=40_000, burn_in=80, n_record=25, seed=11)
    err = sim.freq - exact
    observed = sim.se > 0
    z = err[observed] / sim.se[observed]
    # 41 categories: allow for multiple comparisons with a 4-sigma band.
    assert np.abs(z).max() < 4.0
    inside = np.abs(err) <= 1.96 * sim.se + 1e-15
    assert inside.sum() >= 35
    keep = exact > 0
    _, p = stats.chisquare(sim.final_counts[keep], exact[keep] / exact[keep].sum() * sim.n_chains)
    assert p > 1e-3
    _, df, p_t2 = simulate.hotelling_t2(sim, exact)
    assert df == 39
    assert p_t2 > 1e-3


def test_hotelling_t2_catches_a_small_shift():
    # Make sure the test has teeth: move 0.2 percentage points from Illinois to
    # Boardwalk in the "exact" answer and it should be rejected outright.
    exact = chain.solve("stay").pi
    sim = simulate.simulate("stay", n_chains=40_000, burn_in=80, n_record=25, seed=12)
    wrong = exact.copy()
    wrong[24] -= 0.002
    wrong[39] += 0.002
    assert simulate.hotelling_t2(sim, exact)[2] > 1e-3
    assert simulate.hotelling_t2(sim, wrong)[2] < 1e-6


def test_matches_collins_short_jail_table():
    # Published to 4 decimals in percent, so allow half a unit in the last place plus rounding.
    pi = chain.solve("leave_early").pi * 100
    assert np.abs(pi - np.array(published.SHORT_JAIL)).max() < 1e-4


def test_collins_long_jail_table_assumes_a_reroll_after_jail_doubles():
    table = np.array(published.LONG_JAIL)
    by_the_rules = chain.solve("stay").pi * 100
    with_reroll = chain.solve("stay", jail_doubles_reroll=True).pi * 100
    assert np.abs(by_the_rules - table).max() > 0.05  # In Jail is off by about 0.07 points
    assert np.abs(with_reroll - table).max() < 1e-4


def test_rent_tables_and_costs():
    assert len(board.PROPERTIES) == 22
    for prop in board.PROPERTIES:
        assert list(prop.rents) == sorted(prop.rents)
    brown = economics.group_properties("brown")
    assert sum(p.price for p in brown) + 2 * 3 * 50 == 420


def test_economics_numbers_are_consistent():
    res = chain.solve("stay")
    rows = economics.group_economics(res.per_turn, opponents=3)
    assert [r["group"] for r in rows] == list(board.COLOR_GROUPS)
    for r in rows:
        for lvl in r["levels"]:
            assert lvl["payback_rounds"] * 3 == pytest.approx(lvl["payback_opponent_turns"])
            assert lvl["rent_per_round"] == pytest.approx(3 * lvl["rent_per_opponent_turn"])
    orange = next(r for r in rows if r["group"] == "orange")
    manual = sum(res.per_turn[p.square] * p.rents[3] for p in economics.group_properties("orange"))
    assert orange["levels"][3]["rent_per_opponent_turn"] == pytest.approx(manual)


def test_ranking_does_not_round_twice():
    rows = economics.group_economics(chain.solve("stay").per_turn, opponents=3)
    exact = {r["group"]: r["levels"][5]["payback_rounds"] for r in rows}
    for r in economics.ranking(rows, 5):
        assert f"{r['payback_rounds']:.1f}" == f"{exact[r['group']]:.1f}"
