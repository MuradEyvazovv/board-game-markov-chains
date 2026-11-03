"""Snakes and Ladders on the classic Milton Bradley "Chutes and Ladders" board.

The board has squares 1..100. A token starts off the board (I call that square 0),
moves by one roll of a six-sided die (the original game uses a spinner, same thing),
and follows any ladder or chute it lands on. The game ends on square 100.

Families play the ending in different ways, and the answer depends on it:

* ``"stay"`` (the default here): you must land exactly on 100. If a roll would take you
  past 100, you don't move that turn.
* ``"free_finish"``: any roll that reaches or passes 100 wins.
* ``"bounce"``: you must land exactly on 100, and extra pips count backwards from 100
  (at 98, a roll of 5 takes you to 97).

Square 80 has a ladder straight to 100, so landing on 80 also wins.

Board guides don't all agree on one chute: most list it as 47 -> 26, and some (said to
be newer printings) start it on 48. ``JUMPS_48`` is that variant. Every function that
takes ``jumps`` uses the 47 board unless told otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import markov

LADDERS = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
CHUTES = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
JUMPS = {**LADDERS, **CHUTES}
CHUTES_48 = {(48 if a == 47 else a): b for a, b in CHUTES.items()}
JUMPS_48 = {**LADDERS, **CHUTES_48}
FINISH = 100
N_STATES = FINISH + 1  # squares 0..100
RULES = ("stay", "free_finish", "bounce")


def _target(square: int, roll: int, rule: str, jumps: dict[int, int]) -> int:
    t = square + roll
    if t > FINISH:
        if rule == "stay":
            t = square
        elif rule == "free_finish":
            t = FINISH
        else:
            t = 2 * FINISH - t
    return jumps.get(t, t)


def transition_matrix(rule: str = "stay", jumps: dict[int, int] | None = None) -> np.ndarray:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    jumps = JUMPS if jumps is None else jumps
    P = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        if s == FINISH:
            P[s, s] = 1.0
            continue
        for roll in range(1, 7):
            P[s, _target(s, roll, rule, jumps)] += 1.0 / 6.0
    markov.assert_stochastic(P)
    return P


def transient_states(jumps: dict[int, int] | None = None) -> np.ndarray:
    """Squares a token can sit on before the game ends (jump squares are never rested on)."""
    jumps = JUMPS if jumps is None else jumps
    return np.array([s for s in range(FINISH) if s not in jumps])


@dataclass
class GameLength:
    rule: str
    expected_turns: float
    variance: float
    expected_from_pmf: float
    variance_from_pmf: float
    cdf: np.ndarray  # cdf[t] = P(game over within t turns)
    shortest_game: int
    median: int
    p90: int
    p99: int
    two_player_first_wins: float
    two_player_expected_rounds: float

    @property
    def pmf(self) -> np.ndarray:
        return np.diff(self.cdf, prepend=0.0)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


def _quantile(cdf: np.ndarray, q: float) -> int:
    return int(np.searchsorted(cdf, q - 1e-12))


def analyze(rule: str = "stay", jumps: dict[int, int] | None = None) -> GameLength:
    """Exact game-length results for a single player starting off the board."""
    P = transition_matrix(rule, jumps)
    tr = transient_states(jumps)
    res = markov.absorbing_analysis(P, tr)
    start_row = int(np.where(tr == 0)[0][0])
    expected = float(res["expected_steps"][start_row])
    variance = float(res["variance"][start_row])

    cdf = markov.absorption_cdf(P, start=0, absorbing=FINISH, tail_tol=1e-13)
    pmf = np.diff(cdf, prepend=0.0)
    t = np.arange(len(cdf))
    mean_pmf = float((t * pmf).sum())
    var_pmf = float((t * t * pmf).sum() - mean_pmf**2)

    # Two players taking turns, player 1 first. Player 1 wins if they finish on their
    # t-th turn before player 2 has finished in t - 1 turns.
    survive = 1.0 - cdf  # survive[t] = P(not finished after t turns)
    survive_before = np.concatenate(([1.0], survive[:-1]))
    first_wins = float((pmf * survive_before).sum())
    rounds = float((survive**2).sum())  # E[min(T1, T2)] = sum_t P(both still playing after t)

    return GameLength(
        rule=rule,
        expected_turns=expected,
        variance=variance,
        expected_from_pmf=mean_pmf,
        variance_from_pmf=var_pmf,
        cdf=cdf,
        shortest_game=int(np.argmax(pmf > 0)),
        median=_quantile(cdf, 0.5),
        p90=_quantile(cdf, 0.9),
        p99=_quantile(cdf, 0.99),
        two_player_first_wins=first_wins,
        two_player_expected_rounds=rounds,
    )


def simulate(n_games: int = 1_000_000, seed: int = 2026, rule: str = "stay") -> np.ndarray:
    """Play ``n_games`` independent one-player games; return how many turns each took."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    rng = np.random.default_rng(seed)
    jump = np.arange(FINISH + 1)
    for a, b in JUMPS.items():
        jump[a] = b

    pos = np.zeros(n_games, dtype=np.int64)
    turns = np.zeros(n_games, dtype=np.int64)
    active = np.arange(n_games)
    turn = 0
    while active.size:
        turn += 1
        here = pos[active]
        nxt = here + rng.integers(1, 7, size=active.size)
        if rule == "stay":
            nxt = np.where(nxt > FINISH, here, nxt)
        elif rule == "free_finish":
            nxt = np.minimum(nxt, FINISH)
        else:
            nxt = np.where(nxt > FINISH, 2 * FINISH - nxt, nxt)
        nxt = jump[nxt]
        pos[active] = nxt
        done = nxt == FINISH
        turns[active[done]] = turn
        active = active[~done]
    return turns
