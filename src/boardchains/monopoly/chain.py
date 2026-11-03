"""Exact Markov chain for a single Monopoly token.

One step of the chain is one roll of the dice. The state after a roll is either

* ``(square, streak)``: the token sits on ``square`` and ``streak`` (0, 1 or 2) is the
  number of doubles rolled in a row so far this turn. A streak above 0 means the same
  player rolls again; 0 means the turn is over.
* ``J1``, ``J2``, ``J3``: the token is in jail and its next roll is attempt 1, 2 or 3.

That gives 40 * 3 + 3 = 123 states. A few are never entered (nobody rests on Go To
Jail, and the "leave early" policy never reaches J2 or J3); they just end up with
stationary probability 0.

Rules I model:

* Two six-sided dice. A third double in a row sends you straight to jail without
  moving. Landing on Go To Jail sends you to jail. Going to jail ends your turn.
* Chance and Community Chest are treated as uniform random draws from the full
  16-card deck every time (no memory of which cards are out).
* Chance "Go back 3 spaces" from Chance 3 (square 36) lands on Community Chest 3
  (square 33), and that square's card is drawn and applied too.
* Get Out of Jail Free cards are treated as non-movement cards.

Jail policies:

* ``"leave_early"``: pay $50 on the first jail turn, then roll and move as if starting
  from Just Visiting. Doubles on that roll earn another roll as usual.
* ``"stay"``: try for doubles on each of up to three jail turns. Doubles get you out
  (move by that roll, no extra roll). After a third failed attempt you pay $50 and
  move by that third roll.

The official rules say doubles out of jail don't earn another roll. Some published
tables (Truman Collins', for one) let the player roll again, so ``jail_doubles_reroll``
switches that on for comparison. The escaping double then counts as the first double
of the turn.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .. import markov
from .board import (
    CHANCE_CARDS,
    CHANCE_SQUARES,
    COMMUNITY_CHEST_CARDS,
    COMMUNITY_CHEST_SQUARES,
    GO_TO_JAIL,
    IN_JAIL,
    JUST_VISITING,
    N_CATEGORIES,
    N_SQUARES,
    next_square_in,
)

POLICIES = ("stay", "leave_early")
MAX_STREAK = 3  # streak values 0, 1, 2 are states; the third double goes to jail
J1, J2, J3 = 120, 121, 122
JAIL_STATES = (J1, J2, J3)
N_STATES = N_SQUARES * MAX_STREAK + len(JAIL_STATES)


def state_index(square: int, streak: int) -> int:
    return MAX_STREAK * square + streak


def dice_outcomes() -> list[tuple[int, bool, float]]:
    """All (total, is_double, probability) outcomes of two fair dice, merged by (total, double)."""
    acc: dict[tuple[int, bool], float] = defaultdict(float)
    for a in range(1, 7):
        for b in range(1, 7):
            acc[(a + b, a == b)] += 1.0 / 36.0
    return [(total, dbl, p) for (total, dbl), p in sorted(acc.items())]


def _apply_card(effect: tuple, square: int) -> dict[int, float]:
    kind = effect[0]
    if kind == "square":
        return {effect[1]: 1.0}
    if kind == "nearest":
        return {next_square_in(square, effect[1]): 1.0}
    if kind == "back":
        return resolve_landing((square - effect[1]) % N_SQUARES)
    if kind == "jail":
        return {IN_JAIL: 1.0}
    if kind == "none":
        return {square: 1.0}
    raise ValueError(f"unknown card effect {effect!r}")


@lru_cache(maxsize=None)
def _resolve_landing_cached(square: int) -> tuple[tuple[int, float], ...]:
    if square == GO_TO_JAIL:
        return ((IN_JAIL, 1.0),)
    if square in CHANCE_SQUARES:
        deck = CHANCE_CARDS
    elif square in COMMUNITY_CHEST_SQUARES:
        deck = COMMUNITY_CHEST_CARDS
    else:
        return ((square, 1.0),)
    out: dict[int, float] = defaultdict(float)
    for _, effect in deck:
        for dest, p in _apply_card(effect, square).items():
            out[dest] += p / len(deck)
    return tuple(sorted(out.items()))


def resolve_landing(square: int) -> dict[int, float]:
    """Where a token that the dice put on ``square`` finally ends up.

    Returns ``{destination: probability}`` where a destination is a square 0..39 or
    ``IN_JAIL`` (40). Handles Go To Jail and both card decks, including a chained
    Chance -> Community Chest draw.
    """
    return dict(_resolve_landing_cached(square))


def _add_move(P: np.ndarray, row: int, start: int, total: int, prob: float, streak: int) -> None:
    landing = (start + total) % N_SQUARES
    for dest, q in _resolve_landing_cached(landing):
        col = J1 if dest == IN_JAIL else state_index(dest, streak)
        P[row, col] += prob * q


def transition_matrix(policy: str = "stay", jail_doubles_reroll: bool = False) -> np.ndarray:
    """The 123 x 123 one-roll transition matrix for the given jail policy.

    ``jail_doubles_reroll`` only matters for ``"stay"``: if True, a player who rolls
    doubles to get out of jail rolls again (not the official rule).
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    P = np.zeros((N_STATES, N_STATES))
    dice = dice_outcomes()

    for square in range(N_SQUARES):
        for streak in range(MAX_STREAK):
            row = state_index(square, streak)
            for total, dbl, p in dice:
                if dbl and streak == MAX_STREAK - 1:
                    P[row, J1] += p  # third double in a row: straight to jail, no move
                else:
                    _add_move(P, row, square, total, p, streak + 1 if dbl else 0)

    for attempt, row in enumerate(JAIL_STATES, start=1):
        if policy == "leave_early":
            # Pay before rolling, then it's an ordinary turn from Just Visiting.
            P[row] = P[state_index(JUST_VISITING, 0)]
            continue
        for total, dbl, p in dice:
            if dbl:
                _add_move(P, row, JUST_VISITING, total, p, 1 if jail_doubles_reroll else 0)
            elif attempt == len(JAIL_STATES):
                _add_move(P, row, JUST_VISITING, total, p, 0)
            else:
                P[row, JAIL_STATES[attempt]] += p

    markov.assert_stochastic(P)
    return P


def aggregate(pi_states: np.ndarray) -> np.ndarray:
    """Collapse the 123 states to 41 categories: squares 0..39 plus In Jail (index 40)."""
    out = np.zeros(N_CATEGORIES)
    out[:N_SQUARES] = pi_states[: N_SQUARES * MAX_STREAK].reshape(N_SQUARES, MAX_STREAK).sum(axis=1)
    out[IN_JAIL] = pi_states[list(JAIL_STATES)].sum()
    return out


def turn_end_fraction(pi_states: np.ndarray) -> float:
    """Long-run fraction of rolls that end a turn.

    A roll ends the turn when it leaves the token with streak 0 or in jail. Its
    inverse is the average number of rolls per turn.
    """
    streak0 = pi_states[: N_SQUARES * MAX_STREAK : MAX_STREAK].sum()
    return float(streak0 + pi_states[list(JAIL_STATES)].sum())


@dataclass
class StationaryResult:
    policy: str
    P: np.ndarray
    pi_states: np.ndarray  # from the direct linear solve
    pi: np.ndarray  # 41 categories
    power_iterations: int
    max_diff_power: float
    max_diff_eigen: float
    lambda2: float

    @property
    def rolls_per_turn(self) -> float:
        return 1.0 / turn_end_fraction(self.pi_states)

    @property
    def per_turn(self) -> np.ndarray:
        """Expected landings on each category per player turn (not per roll)."""
        return self.pi / turn_end_fraction(self.pi_states)


def solve(policy: str = "stay", jail_doubles_reroll: bool = False) -> StationaryResult:
    """Build the chain, solve it exactly and cross-check with two other methods."""
    P = transition_matrix(policy, jail_doubles_reroll)
    pi = markov.stationary_linear(P)
    pi_pow, iters = markov.stationary_power(P)
    pi_eig = markov.stationary_eigen(P)
    return StationaryResult(
        policy=policy,
        P=P,
        pi_states=pi,
        pi=aggregate(pi),
        power_iterations=iters,
        max_diff_power=float(np.abs(pi - pi_pow).max()),
        max_diff_eigen=float(np.abs(pi - pi_eig).max()),
        lambda2=markov.second_eigenvalue_modulus(P),
    )
