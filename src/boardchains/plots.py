"""Figures for the README. Static PNGs drawn with matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .monopoly.board import COLOR_GROUPS, N_SQUARES, SQUARE_NAMES, square_kind  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SERIES_3 = "#1baf7a"

# The board's own colors, since they are what people recognize the groups by.
GROUP_COLORS = {
    "brown": "#955436",
    "light blue": "#aae0fa",
    "pink": "#d93a96",
    "orange": "#f7941d",
    "red": "#ed1b24",
    "yellow": "#fef200",
    "green": "#1fb25a",
    "dark blue": "#0072bb",
    "railroad": "#3f3f3f",
    "utility": "#9b9b9b",
    "other": "#d4d3cf",
}

SHORT_NAMES = {
    "Mediterranean Avenue": "Mediterranean",
    "Community Chest 1": "Chest 1",
    "Community Chest 2": "Chest 2",
    "Community Chest 3": "Chest 3",
    "Pennsylvania Railroad": "Penn. RR",
    "Reading Railroad": "Reading RR",
    "B&O Railroad": "B&O RR",
    "Short Line": "Short Line RR",
    "North Carolina Avenue": "N. Carolina",
    "Pennsylvania Avenue": "Pennsylvania",
    "Electric Company": "Electric Co.",
}


def _short(name: str) -> str:
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    return name.replace(" Avenue", "").replace(" Place", " Pl.")


def _color_for_square(sq: int) -> str:
    kind = square_kind(sq)
    return GROUP_COLORS.get(kind, GROUP_COLORS["other"])


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)


def _edge(hex_color: str) -> str:
    """A slightly darker outline so pale fills (yellow, light blue) still read on white."""
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]) * 0.72
    return "#" + "".join(f"{int(c):02x}" for c in rgb)


def landing_chart(pi_stay: np.ndarray, pi_early: np.ndarray, path: Path) -> None:
    """Bars: long-run landing probability per square under the "stay in jail" policy.
    Dashes: the same under "leave jail early"."""
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    x = np.arange(N_SQUARES)
    stay = pi_stay[:N_SQUARES] * 100
    early = pi_early[:N_SQUARES] * 100
    colors = [_color_for_square(s) for s in range(N_SQUARES)]
    ax.bar(x, stay, width=0.78, color=colors, edgecolor=[_edge(c) for c in colors], linewidth=0.7, zorder=2)
    shown = early > 0  # Go To Jail is 0 under both policies; the bar gap says that already
    ax.plot(x[shown], early[shown], linestyle="none", marker="_", markersize=11, markeredgewidth=2.2, color=INK, zorder=3)

    ax.axhline(100 / N_SQUARES, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(7, 100 / N_SQUARES + 0.04, "1/40 = 2.5%", ha="center", va="bottom", fontsize=8.5, color=INK_2)

    top = int(np.argmax(stay))
    ax.annotate(
        f"{SQUARE_NAMES[top]}: {stay[top]:.2f}% (stay) / {early[top]:.2f}% (leave early)",
        xy=(top, max(stay[top], early[top]) + 0.08),
        xytext=(top - 2.5, 3.6),
        fontsize=8.5,
        color=INK,
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8},
    )
    ax.text(30, 0.08, "0%: no roll ends here", rotation=90, ha="center", va="bottom", fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([_short(n) for n in SQUARE_NAMES], rotation=90, fontsize=8.5, color=INK)
    ax.set_xlim(-0.7, N_SQUARES - 0.3)
    ax.set_ylim(0, 3.8)
    ax.set_ylabel("Share of all rolls that end here (%)", fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor=_edge(GROUP_COLORS[g]), label=g) for g in COLOR_GROUPS]
    handles += [
        Patch(facecolor=GROUP_COLORS["railroad"], label="railroad"),
        Patch(facecolor=GROUP_COLORS["utility"], label="utility"),
        Patch(facecolor=GROUP_COLORS["other"], edgecolor=_edge(GROUP_COLORS["other"]), label="other"),
        plt.Line2D([], [], linestyle="none", marker="_", markersize=11, markeredgewidth=2.2, color=INK, label="leave jail early"),
    ]
    leg = ax.legend(
        handles=handles,
        ncol=len(handles),
        fontsize=8.5,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.0),
        handlelength=1.4,
        columnspacing=1.2,
    )
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle(
        "Monopoly: where a token ends up after each roll (bars: stay in jail as long as allowed)",
        x=0.01,
        ha="left",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.925,
        f"Not shown: In Jail, {pi_stay[N_SQUARES] * 100:.2f}% of rolls (stay) or "
        f"{pi_early[N_SQUARES] * 100:.2f}% (leave early). Just Visiting is counted separately.",
        fontsize=9,
        color=INK_2,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def payback_chart(rows: list[dict], path: Path, opponents: int = 3) -> None:
    """Rounds of play to earn back a fully built color group, at 3 houses and at hotels."""
    order = sorted(rows, key=lambda r: r["levels"][3]["payback_rounds"], reverse=True)
    groups = [r["group"] for r in order]
    y = np.arange(len(groups))
    xmax = max(r["levels"][3]["payback_rounds"] for r in rows) * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, level, title in ((axes[0], 3, "3 houses on every lot"), (axes[1], 5, "a hotel on every lot")):
        _style(ax)
        vals = [r["levels"][level]["payback_rounds"] for r in order]
        cols = [GROUP_COLORS[g] for g in groups]
        ax.barh(y, vals, height=0.66, color=cols, edgecolor=[_edge(c) for c in cols], linewidth=0.7, zorder=2)
        for yi, v, r in zip(y, vals, order):
            invest = r["levels"][level]["investment"]
            ax.text(v + xmax * 0.012, yi, f"{v:.1f}  (${invest:,})", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"rounds until rent covers lots + buildings ({opponents} opponents)", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(groups, fontsize=9.5, color=INK)
    fig.suptitle(
        "Monopoly: how fast each color group pays for itself (stay-in-jail landing rates)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.015,
        "Shorter is better. In brackets: total cost of the lots plus buildings. One round = each opponent takes one turn.",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def snakes_chart(results: dict, mc_lengths: np.ndarray, path: Path) -> None:
    """Left: exact distribution of game length vs simulation. Right: CDF under three finishing rules."""
    stay = results["stay"]
    pmf = stay.pmf
    t_max = 160
    t = np.arange(len(pmf))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    _style(ax1)
    hist = np.bincount(mc_lengths, minlength=t_max + 1)[: t_max + 1] / len(mc_lengths)
    ax1.bar(np.arange(t_max + 1), hist * 100, width=1.0, color="#cde2fb", edgecolor="none", zorder=1, label=f"simulation ({len(mc_lengths):,} games)")
    ax1.plot(t[: t_max + 1], pmf[: t_max + 1] * 100, color=SERIES_1, linewidth=2, zorder=3, label="exact (Markov chain)")
    for val in (stay.median, stay.expected_turns):
        ax1.axvline(val, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    peak = pmf.max() * 100
    ax1.text(stay.median - 1, peak * 1.04, f"median {stay.median}", ha="right", fontsize=8.5, color=INK)
    ax1.text(stay.expected_turns + 1, peak * 1.04, f"mean {stay.expected_turns:.2f}", ha="left", fontsize=8.5, color=INK)
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(0, peak * 1.15)
    ax1.set_xlabel("turns to finish (one player)", fontsize=9.5)
    ax1.set_ylabel("probability (%)", fontsize=9.5)
    ax1.set_title("Game length, must land exactly on 100", fontsize=10.5, color=INK, loc="left")
    ax1.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    leg = ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK)

    _style(ax2)
    labels = {"stay": "exact landing (stay put)", "free_finish": "any roll reaching 100 wins", "bounce": "exact landing (bounce back)"}
    colors = {"stay": SERIES_1, "free_finish": SERIES_3, "bounce": SERIES_2}
    for rule in ("free_finish", "stay", "bounce"):
        res = results[rule]
        n = min(len(res.cdf), t_max + 1)
        ax2.plot(np.arange(n), res.cdf[:n], color=colors[rule], linewidth=2, label=f"{labels[rule]}: mean {res.expected_turns:.1f}")
    ax2.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("turns", fontsize=9.5)
    ax2.set_ylabel("P(finished within this many turns)", fontsize=9.5)
    ax2.set_title("Same board, three finishing rules", fontsize=10.5, color=INK, loc="left")
    ax2.grid(True, color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    leg = ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.suptitle("Chutes and Ladders (Milton Bradley board)", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
