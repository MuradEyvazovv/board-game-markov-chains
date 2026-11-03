#!/usr/bin/env python3
"""Rerun the leave-early Monopoly simulation with other seeds.

    python3 scripts/seed_check.py              # same base seed as run_all.py (2026)
    python3 scripts/seed_check.py --seed 7     # if you ran run_all.py with --seed 7

run_all.py simulates the leave-early policy with seed + 1, so 2027 by default. With that
seed, Community Chest 1 came out 4.0 standard errors below the exact value. Across 40
categories that should happen by chance in roughly 1 run out of 400, so this reruns the
same experiment with seeds 1 to 4 to see whether it comes back. Takes about 20 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boardchains.monopoly import board, chain, simulate  # noqa: E402

CHEST_1 = board.SQUARE_NAMES.index("Community Chest 1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026, help="the --seed given to run_all.py")
    args = ap.parse_args()

    exact = chain.solve("leave_early").pi
    rows = []
    for seed in (args.seed + 1, 1, 2, 3, 4):
        sim = simulate.simulate("leave_early", n_chains=1_000_000, burn_in=100, n_record=20, seed=seed)
        z = np.divide(sim.freq - exact, sim.se, out=np.zeros_like(exact), where=sim.se > 0)
        worst = int(np.abs(z).argmax())
        t2, df, p = simulate.hotelling_t2(sim, exact)
        rows.append(
            {
                "seed": seed,
                "z_community_chest_1": round(float(z[CHEST_1]), 3),
                "max_abs_z": round(float(np.abs(z[worst])), 3),
                "max_abs_z_category": board.CATEGORY_NAMES[worst],
                "hotelling_t2": round(t2, 3),
                "hotelling_df": df,
                "hotelling_p": round(p, 4),
            }
        )
        print(rows[-1])
    out = ROOT / "results" / "seed_check.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
