"""Rent income and payback time for each color group, from the exact landing rates.

Setup: I own a complete color group and build evenly. Opponents land on my squares at
the long-run rate from the Markov chain, converted from "per roll" to "per turn"
(a turn can include extra rolls after doubles). One round means every opponent takes
one turn.

Build levels used below: 0 = full set with no houses (rent is doubled by the rules),
1-4 = that many houses on every lot, 5 = a hotel on every lot. A hotel costs one more
house payment on top of four houses, so level ``k`` costs ``k`` house payments per lot.

Payback time = (price of all lots + all building costs) / (expected rent per round).
This ignores mortgages, trading, auctions and the chance that an opponent goes broke
before paying, so read it as a comparison between groups, not a forecast.
"""

from __future__ import annotations

import numpy as np

from .board import COLOR_GROUPS, PROPERTIES, Property

LEVEL_NAMES = ("full set, no houses", "1 house", "2 houses", "3 houses", "4 houses", "hotel")


def rent_at_level(prop: Property, level: int) -> int:
    if level == 0:
        return 2 * prop.rents[0]
    return prop.rents[level]


def group_properties(group: str) -> list[Property]:
    return [p for p in PROPERTIES if p.group == group]


def group_economics(per_turn: np.ndarray, opponents: int = 3) -> list[dict]:
    """One dict per color group with rent, cost and payback at every build level.

    ``per_turn[s]`` is the expected number of times a single opponent lands on square
    ``s`` per turn.
    """
    rows = []
    for group in COLOR_GROUPS:
        props = group_properties(group)
        lots = sum(p.price for p in props)
        house_cost = props[0].house_cost
        landing = {p.name: float(per_turn[p.square]) for p in props}

        levels = []
        for level in range(6):
            rent_per_opp_turn = sum(per_turn[p.square] * rent_at_level(p, level) for p in props)
            invest = lots + level * house_cost * len(props)
            per_round = opponents * rent_per_opp_turn
            levels.append(
                {
                    "level": level,
                    "name": LEVEL_NAMES[level],
                    "investment": invest,
                    "rent_per_opponent_turn": float(rent_per_opp_turn),
                    "rent_per_round": float(per_round),
                    "payback_rounds": float(invest / per_round),
                    "payback_opponent_turns": float(invest / rent_per_opp_turn),
                }
            )

        # Marginal view: what does the k-th round of building buy you?
        marginal = []
        for level in range(1, 6):
            extra_cost = house_cost * len(props)
            extra_rent = levels[level]["rent_per_round"] - levels[level - 1]["rent_per_round"]
            marginal.append(
                {
                    "to_level": level,
                    "name": LEVEL_NAMES[level],
                    "extra_cost": extra_cost,
                    "extra_rent_per_round": float(extra_rent),
                    "payback_rounds": float(extra_cost / extra_rent),
                }
            )

        rows.append(
            {
                "group": group,
                "properties": [p.name for p in props],
                "lot_prices": lots,
                "house_cost": house_cost,
                "landings_per_opponent_turn": landing,
                "group_landings_per_opponent_turn": float(sum(landing.values())),
                "levels": levels,
                "marginal": marginal,
            }
        )
    return rows


def ranking(rows: list[dict], level: int) -> list[dict]:
    """Groups sorted by payback time at one build level (fastest first).

    Values keep 4 decimals. Rounding paybacks to 2 here and then to 1 for display
    turned 9.7491 into 9.75 and then 9.8, so don't round twice.
    """
    ordered = sorted(rows, key=lambda r: r["levels"][level]["payback_rounds"])
    return [
        {
            "rank": i + 1,
            "group": r["group"],
            "investment": r["levels"][level]["investment"],
            "rent_per_opponent_turn": round(r["levels"][level]["rent_per_opponent_turn"], 4),
            "rent_per_round": round(r["levels"][level]["rent_per_round"], 4),
            "payback_rounds": round(r["levels"][level]["payback_rounds"], 4),
            "payback_opponent_turns": round(r["levels"][level]["payback_opponent_turns"], 4),
        }
        for i, r in enumerate(ordered)
    ]
