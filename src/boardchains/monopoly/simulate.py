"""Vectorized Monte Carlo simulation of Monopoly tokens, written from the rules.

This does not touch the transition matrix. It rolls dice and draws cards for many
independent tokens at once with NumPy, so it's an independent check on the exact chain.

Design of the experiment:

* ``n_chains`` independent tokens all start on Go.
* Each one rolls ``burn_in`` times to forget the start, then ``n_record`` more rolls
  whose landing categories are counted.
* Frequencies are pooled over all recorded rolls. Rolls within one token are
  correlated, so the 95% confidence intervals come from the spread of per-token
  frequencies across tokens (which are independent), not from a binomial formula.
* For the chi-square test I only use each token's category after its final roll. That
  gives ``n_chains`` independent draws from the stationary distribution, which is what
  the test assumes. Running it on all the correlated rolls would make the p-value
  meaningless.
* The chi-square test throws away 19 of every 20 recorded rolls, so there's a second
  overall test that uses all of them: Hotelling's T^2 on the per-token frequency
  vectors (see ``hotelling_t2``). Each token's vector is one independent draw, and the
  test estimates the correlation between categories from the data instead of
  assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .board import (
    CHANCE_SQUARES,
    COMMUNITY_CHEST_SQUARES,
    GO_TO_JAIL,
    IN_JAIL,
    JUST_VISITING,
    N_CATEGORIES,
    N_SQUARES,
)

_IS_CHANCE = np.zeros(N_SQUARES, dtype=bool)
_IS_CHANCE[list(CHANCE_SQUARES)] = True
_IS_CHEST = np.zeros(N_SQUARES, dtype=bool)
_IS_CHEST[list(COMMUNITY_CHEST_SQUARES)] = True

# Chance card ids (16 cards, drawn uniformly):
#  0 Go, 1 Illinois Ave, 2 St. Charles Place, 3 nearest Utility, 4-5 nearest Railroad,
#  6 Boardwalk, 7 Reading Railroad, 8 back 3, 9 Go to Jail, 10-15 no movement
# Community Chest card ids: 0 Go, 1 Go to Jail, 2-15 no movement


@dataclass
class SimulationResult:
    policy: str
    n_chains: int
    burn_in: int
    n_record: int
    seed: int
    counts: np.ndarray  # (41,) pooled landing counts over recorded rolls
    freq: np.ndarray  # (41,) pooled frequencies
    se: np.ndarray  # (41,) standard error of freq from between-chain spread
    cov: np.ndarray  # (41, 41) covariance of per-chain frequency vectors
    final_counts: np.ndarray  # (41,) category of each chain after its last roll

    @property
    def total_recorded(self) -> int:
        return self.n_chains * self.n_record

    @property
    def total_rolls(self) -> int:
        return self.n_chains * (self.burn_in + self.n_record)


def _nearest_utility(sq: np.ndarray) -> np.ndarray:
    return np.where((sq < 12) | (sq >= 28), 12, 28)


def _nearest_railroad(sq: np.ndarray) -> np.ndarray:
    return (((sq - 5) // 10 + 1) * 10 + 5) % N_SQUARES


def simulate(
    policy: str = "stay",
    n_chains: int = 1_000_000,
    burn_in: int = 100,
    n_record: int = 10,
    seed: int = 2026,
) -> SimulationResult:
    if policy not in ("stay", "leave_early"):
        raise ValueError(f"unknown policy {policy!r}")
    rng = np.random.default_rng(seed)
    n = n_chains
    idx = np.arange(n)

    pos = np.zeros(n, dtype=np.int64)  # square 0..39 (10 while in jail)
    streak = np.zeros(n, dtype=np.int64)  # doubles in a row this turn
    jail = np.zeros(n, dtype=np.int64)  # 0 = free, k = in jail, next roll is attempt k
    per_chain = np.zeros((n, N_CATEGORIES), dtype=np.int16)
    category = np.zeros(n, dtype=np.int64)

    for step in range(burn_in + n_record):
        d1 = rng.integers(1, 7, size=n)
        d2 = rng.integers(1, 7, size=n)
        total = d1 + d2
        double = d1 == d2

        free = jail == 0
        speeding = free & double & (streak == 2)  # third double: go to jail, don't move
        walk = free & ~speeding

        jailed = ~free
        if policy == "stay":
            leaves = jailed & (double | (jail == 3))
            stuck = jailed & ~leaves
            streak_out = np.zeros(n, dtype=np.int64)  # doubles out of jail: no extra roll
        else:
            leaves = jailed  # paid the fine before rolling
            stuck = np.zeros(n, dtype=bool)
            streak_out = double.astype(np.int64)  # a normal turn, so doubles roll again

        movers = walk | leaves
        start = np.where(leaves, JUST_VISITING, pos)
        landing = (start + total) % N_SQUARES
        new_streak = np.where(leaves, streak_out, np.where(double, streak + 1, 0))

        final = landing.copy()
        to_jail = speeding | (movers & (landing == GO_TO_JAIL))

        on_chance = movers & _IS_CHANCE[landing]
        card = rng.integers(0, 16, size=n)
        final = np.where(on_chance & (card == 0), 0, final)
        final = np.where(on_chance & (card == 1), 24, final)
        final = np.where(on_chance & (card == 2), 11, final)
        final = np.where(on_chance & (card == 3), _nearest_utility(landing), final)
        final = np.where(on_chance & ((card == 4) | (card == 5)), _nearest_railroad(landing), final)
        final = np.where(on_chance & (card == 6), 39, final)
        final = np.where(on_chance & (card == 7), 5, final)
        final = np.where(on_chance & (card == 8), landing - 3, final)
        to_jail |= on_chance & (card == 9)

        # Community Chest, including tokens sent there by "go back 3" from square 36.
        on_chest = movers & ~to_jail & _IS_CHEST[final]
        card = rng.integers(0, 16, size=n)
        final = np.where(on_chest & (card == 0), 0, final)
        to_jail |= on_chest & (card == 1)

        pos = np.where(movers, final, pos)
        streak = np.where(movers, new_streak, streak)
        jail = np.where(leaves, 0, np.where(stuck, jail + 1, jail))
        jail[to_jail] = 1
        streak[to_jail] = 0
        pos[to_jail] = JUST_VISITING

        category = np.where(jail > 0, IN_JAIL, pos)
        if step >= burn_in:
            per_chain[idx, category] += 1

    counts = per_chain.sum(axis=0, dtype=np.int64)
    freq = counts / (n * n_record)
    # Sample covariance of the per-chain frequency vectors. X^T X is built in chunks so
    # the int16 count matrix never has to be copied to float64 all at once.
    xtx = np.zeros((N_CATEGORIES, N_CATEGORIES))
    for lo in range(0, n, 200_000):
        block = per_chain[lo : lo + 200_000].astype(np.float64)
        xtx += block.T @ block
    cov = (xtx / n_record**2 - n * np.outer(freq, freq)) / (n - 1)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0) / n)  # SE of the pooled frequency
    final_counts = np.bincount(category, minlength=N_CATEGORIES)

    return SimulationResult(
        policy=policy,
        n_chains=n,
        burn_in=burn_in,
        n_record=n_record,
        seed=seed,
        counts=counts,
        freq=freq,
        se=se,
        cov=cov,
        final_counts=final_counts,
    )


def hotelling_t2(sim: SimulationResult, exact: np.ndarray) -> tuple[float, int, float]:
    """Test whether the simulated frequencies match ``exact`` in all categories at once.

    Each token gives one frequency vector, and tokens are independent, so with ``n``
    tokens ``T^2 = n d' S^-1 d`` is approximately chi-square under the null. Here
    ``d`` is the pooled frequency minus the exact value and ``S`` is the sample
    covariance of the per-token vectors. Each vector sums to 1, which makes ``S``
    singular, so I drop categories with exact probability 0 (Go To Jail) and then the
    last remaining one. Which one gets dropped doesn't change ``T^2``.

    Returns ``(T^2, degrees of freedom, p-value)``.
    """
    keep = np.flatnonzero(exact > 0)[:-1]
    d = sim.freq[keep] - exact[keep]
    S = sim.cov[np.ix_(keep, keep)]
    t2 = float(sim.n_chains * d @ np.linalg.solve(S, d))
    df = len(keep)
    return t2, df, float(stats.chi2.sf(t2, df))
