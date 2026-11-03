"""Generic Markov chain tools shared by the Monopoly and Snakes and Ladders models.

Everything here works on a dense row-stochastic matrix ``P`` where ``P[i, j]`` is the
probability of moving from state ``i`` to state ``j`` in one step.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def assert_stochastic(P: np.ndarray, atol: float = 1e-12) -> None:
    """Raise ``ValueError`` unless ``P`` is square, non-negative and every row sums to 1."""
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"transition matrix must be square, got shape {P.shape}")
    if (P < -atol).any():
        raise ValueError("transition matrix has negative entries")
    row_sums = P.sum(axis=1)
    worst = np.abs(row_sums - 1.0).max()
    if worst > atol:
        bad = int(np.abs(row_sums - 1.0).argmax())
        raise ValueError(f"row {bad} sums to {row_sums[bad]!r} (off by {worst:.3e})")


def stationary_linear(P: np.ndarray) -> np.ndarray:
    """Solve ``pi (P - I) = 0`` with ``sum(pi) = 1`` directly.

    The n balance equations are linearly dependent (they add up to 0 = 0), so I drop
    the last one and put the normalization constraint in its place. The resulting
    system is non-singular whenever the chain has a single closed class, which is
    true for both games here.
    """
    n = P.shape[0]
    A = (P - np.eye(n)).T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = linalg.solve(A, b)
    pi[np.abs(pi) < 1e-15] = 0.0
    return pi


def stationary_power(
    P: np.ndarray, tol: float = 1e-14, max_iter: int = 100_000
) -> tuple[np.ndarray, int]:
    """Power iteration from the uniform distribution. Returns ``(pi, iterations)``."""
    n = P.shape[0]
    pi = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        nxt = pi @ P
        if np.abs(nxt - pi).sum() < tol:
            return nxt / nxt.sum(), it
        pi = nxt
    raise RuntimeError(f"power iteration did not converge in {max_iter} steps")


def stationary_eigen(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of ``P`` for the eigenvalue closest to 1, scaled to sum to 1."""
    w, v = linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    vec = np.real(v[:, k])
    vec = vec / vec.sum()
    vec[np.abs(vec) < 1e-15] = 0.0
    return vec


def second_eigenvalue_modulus(P: np.ndarray) -> float:
    """``|lambda_2|``: how fast the chain forgets its starting point (smaller is faster)."""
    w = np.sort(np.abs(linalg.eigvals(P)))[::-1]
    return float(w[1])


def absorbing_analysis(P: np.ndarray, transient: np.ndarray) -> dict[str, np.ndarray]:
    """Fundamental-matrix results for an absorbing chain.

    ``transient`` lists the indices of the transient states. Returns the fundamental
    matrix ``N = (I - Q)^-1``, the expected number of steps to absorption from each
    transient state (``t = N 1``) and its variance (``(2N - I) t - t*t``).
    """
    transient = np.asarray(transient)
    Q = P[np.ix_(transient, transient)]
    eye = np.eye(len(transient))
    N = linalg.inv(eye - Q)
    t = N.sum(axis=1)
    var = (2.0 * N - eye) @ t - t * t
    return {"N": N, "expected_steps": t, "variance": var}


def absorption_cdf(
    P: np.ndarray,
    start: int,
    absorbing: int,
    tail_tol: float = 1e-13,
    max_steps: int = 100_000,
) -> np.ndarray:
    """``cdf[t] = P(absorbed by step t)`` for ``t = 0, 1, ...`` until the tail is below ``tail_tol``."""
    dist = np.zeros(P.shape[0])
    dist[start] = 1.0
    others = np.arange(P.shape[0]) != absorbing
    cdf = [dist[absorbing]]
    for _ in range(max_steps):
        dist = dist @ P
        cdf.append(dist[absorbing])
        # Watch the mass still in play rather than 1 - cdf, which stalls at rounding error.
        if dist[others].sum() < tail_tol:
            return np.array(cdf)
    raise RuntimeError("absorption_cdf hit max_steps before the tail got small enough")
