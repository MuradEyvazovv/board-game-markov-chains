#!/usr/bin/env python3
"""Reproduce every number and figure in the README.

    python3 scripts/run_all.py            # full run (about 10 s on a laptop)
    python3 scripts/run_all.py --quick    # smaller simulations, for a fast check

Writes JSON to results/ and PNGs to figures/.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

from boardchains import plots, snakes  # noqa: E402
from boardchains.monopoly import board, chain, economics, published, simulate  # noqa: E402


def _r(x: float, digits: int = 8) -> float:
    return float(f"{x:.{digits}g}")


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def monopoly_exact() -> dict:
    out = {}
    for policy in chain.POLICIES:
        res = chain.solve(policy)
        order = np.argsort(-res.pi[: board.N_SQUARES])
        out[policy] = {
            "result": res,
            "json": {
                "policy": policy,
                "in_jail": _r(res.pi[board.IN_JAIL]),
                "just_visiting": _r(res.pi[board.JUST_VISITING]),
                "rolls_per_turn": _r(res.rolls_per_turn),
                "top10_squares": [
                    {"rank": i + 1, "square": int(s), "name": board.SQUARE_NAMES[s], "probability": _r(res.pi[s])}
                    for i, s in enumerate(order[:10])
                ],
                "all_categories": [
                    {
                        "index": i,
                        "name": board.CATEGORY_NAMES[i],
                        "kind": board.square_kind(i),
                        "probability_per_roll": _r(res.pi[i]),
                        "landings_per_turn": _r(res.per_turn[i]),
                    }
                    for i in range(board.N_CATEGORIES)
                ],
                "solver_checks": {
                    "method": "direct linear solve of pi(P - I) = 0 with sum(pi) = 1",
                    "n_states": int(res.P.shape[0]),
                    "max_row_sum_error": _r(float(np.abs(res.P.sum(axis=1) - 1).max()), 3),
                    "stationary_sum": _r(float(res.pi_states.sum()), 15),
                    "fixed_point_error": _r(float(np.abs(res.pi_states @ res.P - res.pi_states).max()), 3),
                    "max_diff_vs_power_iteration": _r(res.max_diff_power, 3),
                    "power_iterations": res.power_iterations,
                    "max_diff_vs_eigenvector": _r(res.max_diff_eigen, 3),
                    "second_eigenvalue_modulus": _r(res.lambda2, 6),
                },
            },
        }
    return out


def published_check(exact: dict) -> dict:
    """Compare the exact chain with Truman Collins' published table (percent, 4 decimals)."""
    reroll = chain.solve("stay", jail_doubles_reroll=True).pi
    pairs = {
        "short_jail_vs_leave_early": (published.SHORT_JAIL, exact["leave_early"]["result"].pi),
        "long_jail_vs_stay": (published.LONG_JAIL, exact["stay"]["result"].pi),
        "long_jail_vs_stay_with_reroll_after_jail_doubles": (published.LONG_JAIL, reroll),
    }
    out = {
        "source": "Truman Collins, Probabilities in the Game of Monopoly, tkcs-collins.com (first table)",
        "note": "Collins' long-jail column only matches if doubles out of jail earn another roll, "
        "which the official rules don't allow; the 'stay' policy here follows the rules.",
    }
    for key, (table, pi) in pairs.items():
        diff = np.abs(np.array(table) - 100 * pi)
        worst = int(diff.argmax())
        out[key] = {
            "max_abs_diff_percentage_points": _r(float(diff.max()), 3),
            "worst_category": board.CATEGORY_NAMES[worst],
            "categories_within_0.0001": f"{int((diff <= 1e-4).sum())} of {len(diff)}",
        }
    return out


def monopoly_mc(exact: dict, n_chains: int, burn_in: int, n_record: int, seed: int) -> dict:
    out = {}
    for k, policy in enumerate(chain.POLICIES):
        t0 = time.perf_counter()
        sim = simulate.simulate(policy, n_chains=n_chains, burn_in=burn_in, n_record=n_record, seed=seed + k)
        secs = time.perf_counter() - t0
        pi = exact[policy]["result"].pi
        err = sim.freq - pi
        z = np.divide(err, sim.se, out=np.zeros_like(err), where=sim.se > 0)
        lo, hi = sim.freq - 1.96 * sim.se, sim.freq + 1.96 * sim.se
        inside = (pi >= lo) & (pi <= hi)
        keep = pi > 0  # Go To Jail can't be observed, so it's left out of both checks
        expected = pi[keep] / pi[keep].sum() * sim.n_chains
        chi2, p = stats.chisquare(sim.final_counts[keep], expected)
        t2, t2_df, t2_p = simulate.hotelling_t2(sim, pi)
        worst = int(np.abs(err).argmax())
        lam2 = exact[policy]["result"].lambda2

        out[policy] = {
            "policy": policy,
            "seed": seed + k,
            "n_chains": sim.n_chains,
            "burn_in_rolls_per_chain": sim.burn_in,
            "recorded_rolls_per_chain": sim.n_record,
            "total_rolls_simulated": sim.total_rolls,
            "total_rolls_recorded": sim.total_recorded,
            "start_bias_bound_lambda2_pow_burn_in": _r(lam2**burn_in, 3),
            "runtime_seconds": round(secs, 2),
            "max_abs_error": _r(float(np.abs(err).max()), 4),
            "max_abs_error_category": board.CATEGORY_NAMES[worst],
            "max_abs_z": _r(float(np.abs(z).max()), 4),
            "categories_with_exact_inside_95ci": f"{int(inside[keep].sum())} of {int(keep.sum())}",
            "chi_square": {
                "data": "category of each chain after its last roll (independent draws)",
                "n_draws": sim.n_chains,
                "statistic": _r(float(chi2), 6),
                "df": int(keep.sum() - 1),
                "p_value": _r(float(p), 4),
            },
            "hotelling_t2": {
                "data": "per-chain frequency vectors over all recorded rolls (independent across chains)",
                "n_vectors": sim.n_chains,
                "statistic": _r(t2, 6),
                "df": t2_df,
                "p_value": _r(t2_p, 4),
            },
            "per_category": [
                {
                    "name": board.CATEGORY_NAMES[i],
                    "exact": _r(pi[i]),
                    "simulated": _r(sim.freq[i]),
                    "z": _r(float(z[i]), 4),
                    "ci95_low": _r(lo[i]),
                    "ci95_high": _r(hi[i]),
                }
                for i in range(board.N_CATEGORIES)
            ],
        }
    return out


def snakes_all(n_games: int, seed: int) -> tuple[dict, dict, np.ndarray]:
    exact = {rule: snakes.analyze(rule) for rule in snakes.RULES}
    alt = snakes.analyze("stay", snakes.JUMPS_48)
    t0 = time.perf_counter()
    lengths = snakes.simulate(n_games, seed=seed, rule="stay")
    secs = time.perf_counter() - t0
    stay = exact["stay"]

    mean = float(lengths.mean())
    se = float(lengths.std(ddof=1) / np.sqrt(n_games))
    ecdf = np.searchsorted(np.sort(lengths), np.arange(len(stay.cdf)), side="right") / n_games
    j = {
        "board": {"ladders": snakes.LADDERS, "chutes": snakes.CHUTES},
        "default_rule": "stay: you must land exactly on 100; a roll that would overshoot means you don't move",
        "chute_48_variant_stay_rule": {
            "change": "chute 47->26 moved to 48->26, everything else the same",
            "expected_turns": _r(alt.expected_turns, 10),
            "variance": _r(alt.variance, 10),
            "median": alt.median,
        },
        "exact": {
            rule: {
                "expected_turns": _r(g.expected_turns, 10),
                "variance": _r(g.variance, 10),
                "std": _r(g.std, 8),
                "expected_from_distribution": _r(g.expected_from_pmf, 10),
                "median": g.median,
                "p90": g.p90,
                "p99": g.p99,
                "shortest_game": g.shortest_game,
                "p_shortest_game": _r(float(g.pmf[g.shortest_game]), 6),
                "p_done_within_10_turns": _r(float(g.cdf[10]), 6),
                "p_longer_than_100_turns": _r(float(1 - g.cdf[100]), 6),
                "two_player_p_first_player_wins": _r(g.two_player_first_wins, 6),
                "two_player_expected_rounds": _r(g.two_player_expected_rounds, 6),
            }
            for rule, g in exact.items()
        },
        "monte_carlo_stay_rule": {
            "n_games": n_games,
            "seed": seed,
            "runtime_seconds": round(secs, 2),
            "mean": _r(mean, 8),
            "mean_ci95": [_r(mean - 1.96 * se, 8), _r(mean + 1.96 * se, 8)],
            "z_vs_exact": _r((mean - stay.expected_turns) / se, 4),
            "variance": _r(float(lengths.var(ddof=1)), 8),
            # Same definition as the exact median: smallest t with P(T <= t) >= 0.5.
            "median": int(np.quantile(lengths, 0.5, method="inverted_cdf")),
            "median_note": "exact P(T <= 32) is 0.50005, so the simulated median is 31 or 32 "
            "almost at random; it isn't a useful check",
            "shortest": int(lengths.min()),
            "longest": int(lengths.max()),
            "max_abs_cdf_difference": _r(float(np.abs(ecdf - stay.cdf).max()), 4),
        },
        "cdf_stay_rule_first_200_turns": [_r(float(x), 10) for x in stay.cdf[:201]],
    }
    return exact, j, lengths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller simulations for a fast check")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    n_chains, burn_in, n_record, n_games = 1_000_000, 100, 20, 1_000_000
    if args.quick:
        n_chains, n_record, n_games = 100_000, 10, 100_000

    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    timings = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    exact = monopoly_exact()
    timings["monopoly_exact"] = time.perf_counter() - t0
    exact_json = {p: exact[p]["json"] for p in chain.POLICIES}
    exact_json["published_comparison"] = published_check(exact)
    _dump(results / "monopoly_exact.json", exact_json)

    t0 = time.perf_counter()
    mc = monopoly_mc(exact, n_chains, burn_in, n_record, args.seed)
    timings["monopoly_monte_carlo"] = time.perf_counter() - t0
    _dump(results / "monopoly_monte_carlo.json", mc)

    econ = {}
    for policy in chain.POLICIES:
        rows = economics.group_economics(exact[policy]["result"].per_turn, opponents=3)
        econ[policy] = {
            "opponents": 3,
            "ranking_3_houses": economics.ranking(rows, 3),
            "ranking_hotel": economics.ranking(rows, 5),
            "best_marginal_build_step": {
                r["group"]: min(r["marginal"], key=lambda m: m["payback_rounds"])["name"] for r in rows
            },
            "groups": rows,
        }
    _dump(results / "monopoly_economics.json", econ)

    t0 = time.perf_counter()
    sl_exact, sl_json, lengths = snakes_all(n_games, args.seed)
    timings["snakes_and_ladders"] = time.perf_counter() - t0
    _dump(results / "snakes_and_ladders.json", sl_json)

    t0 = time.perf_counter()
    plots.landing_chart(exact["stay"]["result"].pi, exact["leave_early"]["result"].pi, figures / "monopoly_landing_probabilities.png")
    stay_rows = econ["stay"]["groups"]
    plots.payback_chart(stay_rows, figures / "monopoly_payback_by_group.png")
    plots.snakes_chart(sl_exact, lengths, figures / "snakes_game_length.png")
    timings["figures"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_all

    summary = {
        "mode": "quick" if args.quick else "full",
        "seed": args.seed,
        "runtime_seconds": {k: round(v, 2) for k, v in timings.items()},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "machine": platform.machine(),
        },
        "monopoly": {
            p: {
                "in_jail": exact[p]["json"]["in_jail"],
                "top10": [(e["name"], e["probability"]) for e in exact[p]["json"]["top10_squares"]],
                "mc_max_abs_error": mc[p]["max_abs_error"],
                "mc_chi_square_p": mc[p]["chi_square"]["p_value"],
                "mc_hotelling_t2_p": mc[p]["hotelling_t2"]["p_value"],
                "mc_inside_95ci": mc[p]["categories_with_exact_inside_95ci"],
                "ranking_3_houses": [r["group"] for r in econ[p]["ranking_3_houses"]],
                "ranking_hotel": [r["group"] for r in econ[p]["ranking_hotel"]],
            }
            for p in chain.POLICIES
        },
        "snakes_and_ladders": {
            "expected_turns": sl_json["exact"]["stay"]["expected_turns"],
            "variance": sl_json["exact"]["stay"]["variance"],
            "median": sl_json["exact"]["stay"]["median"],
            "mc_mean": sl_json["monte_carlo_stay_rule"]["mean"],
            "mc_mean_ci95": sl_json["monte_carlo_stay_rule"]["mean_ci95"],
            "chute_48_variant_expected_turns": sl_json["chute_48_variant_stay_rule"]["expected_turns"],
        },
    }
    _dump(results / "summary.json", summary)

    # Console report
    print(f"Mode: {summary['mode']}, seed {args.seed}")
    for p in chain.POLICIES:
        e, m = exact[p]["json"], mc[p]
        print(f"\nMonopoly, jail policy '{p}': In Jail {e['in_jail']:.4%}, {e['rolls_per_turn']:.4f} rolls per turn")
        for row in e["top10_squares"]:
            print(f"  {row['rank']:>2}. {row['name']:<24} {row['probability']:.4%}")
        print(
            f"  Monte Carlo: {m['total_rolls_recorded']:,} recorded rolls, max abs error {m['max_abs_error']:.2e} "
            f"({m['max_abs_error_category']}), exact inside 95% CI for {m['categories_with_exact_inside_95ci']}, "
            f"chi-square p = {m['chi_square']['p_value']:.3f} (df {m['chi_square']['df']}), "
            f"Hotelling T^2 = {m['hotelling_t2']['statistic']:.2f}, p = {m['hotelling_t2']['p_value']:.3f}"
        )
        print("  Payback at 3 houses (rounds, 3 opponents): " + ", ".join(
            f"{r['group']} {r['payback_rounds']:.1f}" for r in econ[p]["ranking_3_houses"]))
        print("  Payback at hotel (rounds, 3 opponents):    " + ", ".join(
            f"{r['group']} {r['payback_rounds']:.1f}" for r in econ[p]["ranking_hotel"]))
    s = sl_json["exact"]["stay"]
    m = sl_json["monte_carlo_stay_rule"]
    print(
        f"\nChutes and Ladders (exact landing on 100): E[T] = {s['expected_turns']:.4f}, Var = {s['variance']:.4f}, "
        f"SD = {s['std']:.4f}, median {s['median']}, shortest {s['shortest_game']}"
    )
    print(f"  Monte Carlo ({m['n_games']:,} games): mean {m['mean']:.4f}, 95% CI {m['mean_ci95']}, median {m['median']}")
    for rule in ("free_finish", "bounce"):
        print(f"  Rule '{rule}': E[T] = {sl_json['exact'][rule]['expected_turns']:.4f}")
    print(f"  Chute at 48 instead of 47: E[T] = {sl_json['chute_48_variant_stay_rule']['expected_turns']:.4f}")
    pc = exact_json["published_comparison"]
    print("\nCollins' published table, max gap in percentage points:")
    for key in ("short_jail_vs_leave_early", "long_jail_vs_stay", "long_jail_vs_stay_with_reroll_after_jail_doubles"):
        print(f"  {key}: {pc[key]['max_abs_diff_percentage_points']:.2g} ({pc[key]['worst_category']})")
    print(f"\nTotal runtime: {timings['total']:.1f} s")


if __name__ == "__main__":
    main()
