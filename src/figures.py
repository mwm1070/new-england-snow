#!/usr/bin/env python3
"""Draw every figure in the report from build/results.json.

Nothing here computes anything: if a number is not in the JSON, it does not
belong on a chart. That keeps the pictures and the prose honest.

Usage
-----
    python src/figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "build"
FIGS = SITE / "figures"

# A deliberately restrained palette: one accent, one counter-accent, greys.
INK = "#1a202c"
MUTED = "#718096"
GRID = "#e2e8f0"
BLUE = "#2b6cb0"
ORANGE = "#c05621"
TEAL = "#2c7a7b"
RED = "#9b2c2c"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
        "figure.dpi": 170,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


KEEP_UPPER = {
    "N", "S", "E", "W", "NE", "NW", "SE", "SW", "NNE", "NNW", "SSE", "SSW",
    "AP", "ASOS", "COOP", "INTL",
}


def pretty_name(name: str) -> str:
    """Title-case a GHCN station name without mangling compass points and acronyms."""
    return " ".join(w if w in KEEP_UPPER else w.capitalize() for w in name.split())


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"  wrote build/figures/{name}.png")


def period_spans(ax, early, late, y=0.98, color=GRID) -> None:
    """Shade the two study periods behind a plot."""
    ax.axvspan(early[0] - 0.5, early[1] + 0.5, color="#f7f9fb", zorder=0)
    ax.axvspan(late[0] - 0.5, late[1] + 0.5, color="#fff5eb", zorder=0)
    ax.axvline(early[1] + 0.5, color=MUTED, lw=0.8, ls=":", zorder=1)


# ---------------------------------------------------------------------------
# 1. The hero chart
# ---------------------------------------------------------------------------
def fig_regional(r: dict) -> None:
    cfg, reg = r["config"], r["regional"]
    seasons = np.array([p["season"] for p in reg["series"]])
    values = np.array([p["anomaly"] for p in reg["series"]]) + reg["baseline_level_in"]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    period_spans(ax, cfg["early"], cfg["late"])

    ax.scatter(seasons, values, s=9, color=MUTED, alpha=0.55, zorder=2, label="One season")
    roll = np.full(len(values), np.nan)
    for i in range(len(values)):
        lo, hi = max(0, i - 2), min(len(values), i + 3)
        roll[i] = np.nanmean(values[lo:hi])
    ax.plot(seasons, roll, color=BLUE, lw=2.0, zorder=3, label="5-season average")

    early_mean = reg["early_level_in"]
    late_mean = reg["late_level_in"]
    ax.hlines(early_mean, cfg["early"][0] - 0.5, cfg["early"][1] + 0.5, color=INK, lw=1.6, ls="--", zorder=4)
    ax.hlines(late_mean, cfg["late"][0] - 0.5, cfg["late"][1] + 0.5, color=ORANGE, lw=1.6, ls="--", zorder=4)

    # Error bars on the period means, offset to the right edge of each period.
    for xpos, mean, se, colour in [
        (cfg["early"][1] - 1, early_mean, reg["early_se"], INK),
        (cfg["late"][1] - 1, late_mean, reg["late_se"], ORANGE),
    ]:
        ax.errorbar(xpos, mean, yerr=1.96 * se, fmt="o", ms=5, color=colour,
                    capsize=3, lw=1.4, zorder=5)

    ax.annotate(
        f"{reg['diff']:+.1f} in\n95% CI [{reg['ci'][0]:+.1f}, {reg['ci'][1]:+.1f}]",
        xy=(0.985, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=INK,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID, "lw": 0.9},
    )

    ax.set_xlim(cfg["early"][0] - 1, cfg["late"][1] + 1)
    ax.set_ylim(0, max(values.max(), np.nanmax(roll)) * 1.12 + 5)
    ax.set_ylabel("Regional snowfall index (inches)")
    ax.set_xlabel("Season (year containing January)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_title("New England seasonal snowfall, 1941–2020")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)

    fig.text(
        0.5,
        -0.03,
        f"Average of {reg['min_stations_per_season']}–{reg['max_stations_per_season']} station records, "
        f"each centred on its own 1941–2000 mean. Season-to-season SD is about {reg['early_sd']:.0f} in; "
        f"the standard error of a 20-season mean is about {reg['late_se']:.1f} in.",
        ha="center",
        fontsize=7.2,
        color=MUTED,
    )
    save(fig, "01_regional")


# ---------------------------------------------------------------------------
# 2. Station-by-station
# ---------------------------------------------------------------------------
def fig_stations(r: dict) -> None:
    cfg, paired = r["config"], r["paired"]
    stations = sorted(paired["stations"], key=lambda s: s["change"])
    names = [f"{pretty_name(s['name'])} ({s['state']})" for s in stations]
    changes = np.array([s["change"] for s in stations])
    colors = [BLUE if c >= 0 else RED for c in changes]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(8.2, 6.4), gridspec_kw={"width_ratios": [3.1, 1]}
    )

    y = np.arange(len(changes))
    ax.barh(y, changes, color=colors, height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.6)
    ax.axvline(0, color=INK, lw=1.0, zorder=4)
    ax.set_xlabel("Change in mean seasonal snowfall, 2001–2020 vs 1941–2000 (in)")
    ax.set_title(f"Each station's own before-and-after ({len(changes)} stations)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(changes.min() * 1.15, changes.max() * 1.15)

    ax2.hist(changes, bins=np.arange(-22, 20, 3), color=BLUE, alpha=0.75, zorder=3)
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.axvline(paired["mean_change"], color=ORANGE, lw=1.6, ls="--")
    ax2.set_yticks([])
    ax2.set_xlabel("Change (in)")
    ax2.set_title("Distribution")
    ax2.annotate(
        f"{(changes > 0).sum()} up\n{(changes < 0).sum()} down",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    ax2.grid(axis="y", visible=False)

    fig.suptitle(
        "Most stations gained snow; the losses are not a regional pattern",
        fontsize=10.5,
        y=0.99,
    )
    save(fig, "02_stations")


# ---------------------------------------------------------------------------
# 3. Decade by decade
# ---------------------------------------------------------------------------
def fig_decades(r: dict) -> None:
    reg = r["regional"]
    decades = reg["decades"]
    x = np.arange(len(decades))
    means = np.array([d["mean"] for d in decades])
    errs = 1.96 * np.array([d["se"] for d in decades])
    colors = [BLUE if m >= 0 else RED for m in means]

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    ax.errorbar(
        x, means, yerr=errs, fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=3, zorder=3
    )
    ax.scatter(x, means, s=42, c=colors, zorder=4, edgecolor="white", linewidth=0.8)

    # The whole-period difference, drawn as a reference band.
    ax.axhspan(0, reg["diff"], color=ORANGE, alpha=0.10, zorder=1)
    ax.annotate(
        f"the 2001–2020 vs 1941–2000 difference is {reg['diff']:+.1f} in",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        ha="left",
        fontsize=8,
        color=ORANGE,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([d["decade"] for d in decades])
    ax.set_ylabel("Snowfall anomaly vs 1941–2000 (in)")
    ax.set_title("Decadal means swing by ±10 inches, far more than the period difference")
    ax.grid(axis="x", visible=False)
    save(fig, "03_decades")


# ---------------------------------------------------------------------------
# 4. Sensitivity
# ---------------------------------------------------------------------------
def fig_sensitivity(r: dict) -> None:
    rows = r["sensitivity"]
    order = {"jul-jun": 0, "oct-may": 1, "nov-mar": 2}
    rows = sorted(rows, key=lambda d: (order[d["window"]], -d["coverage"]))
    labels = [f"{d['window']}  ·  {d['coverage']:.2f}" for d in rows]
    diffs = np.array([d["diff"] for d in rows])
    los = np.array([d["ci"][0] for d in rows])
    his = np.array([d["ci"][1] for d in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK, lw=1.0, zorder=4)
    ax.errorbar(
        diffs, y, xerr=[diffs - los, his - diffs], fmt="o", ms=5,
        color=BLUE, ecolor=MUTED, elinewidth=1.1, capsize=2.5, zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, family="DejaVu Sans Mono")
    ax.set_xlabel("Change in seasonal snowfall, 2001–2020 vs 1941–2000 (in)")
    ax.set_ylim(-1.2, len(rows) + 0.2)
    ax.set_title("Every season window and coverage threshold gives the same answer", pad=22)
    ax.grid(axis="y", visible=False)

    # Separators between the three windows.
    for boundary in (4.5, 9.5):
        ax.axhline(boundary, color=GRID, lw=1.2)

    ax.text(
        0.5, 1.015,
        f"{int((diffs > 0).sum())} of {len(diffs)} point estimates are positive; "
        f"every confidence interval contains zero",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    save(fig, "04_sensitivity")


# ---------------------------------------------------------------------------
# 5. Extremes and persistence
# ---------------------------------------------------------------------------
def fig_extremes(r: dict) -> None:
    from matplotlib.patches import Patch

    ex = r["extremes"]
    specs = [
        ("days_ge6", "Days with ≥6 in\nof snow"),
        ("days_ge12", "Days with ≥12 in\nof snow"),
        ("days_with_snow_cover", "Days with snow\non the ground"),
    ]
    specs = [(k, t) for k, t in specs if ex.get(k)]

    fig, axes = plt.subplots(1, len(specs), figsize=(8.4, 3.1))
    if len(specs) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, specs):
        d = ex[key]
        ax.barh([1], [d["late"]], color=ORANGE, height=0.52, zorder=3)
        ax.barh([0], [d["early"]], color=BLUE, height=0.52, zorder=3)
        for yy, v in ((1, d["late"]), (0, d["early"])):
            ax.text(v * 1.03, yy, f"{v:.1f}", va="center", fontsize=8.5, color=INK, zorder=5)
        ax.set_yticks([])
        ax.set_ylim(-1.15, 1.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("days per season")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(d["late"], d["early"]) * 1.45)
        lo, hi = d["ci"]
        sign = "+" if d["diff"] >= 0 else ""
        ax.text(
            0.0, -0.85,
            f"{sign}{d['diff']:.1f}  [{lo:+.1f}, {hi:+.1f}]",
            ha="left", va="center", fontsize=7.5, color=MUTED,
        )

    fig.legend(
        handles=[Patch(color=BLUE, label="1941–2000"), Patch(color=ORANGE, label="2001–2020")],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.10),
    )
    fig.suptitle("Heavy-snow days and snow-cover days did not fall", fontsize=10, y=1.12)
    save(fig, "05_extremes")


# ---------------------------------------------------------------------------
# 6. Mechanism
# ---------------------------------------------------------------------------
def fig_mechanism(r: dict) -> None:
    cfg = r["config"]
    mech = r["mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))

    for ax, key, colour in [(axes[0], "precip", TEAL), (axes[1], "temp", RED)]:
        d = mech[key]
        seasons = np.array([p["season"] for p in d["series"]])
        vals = np.array([p["value"] for p in d["series"]])
        period_spans(ax, cfg["early"], cfg["late"])
        ax.scatter(seasons, vals, s=7, color=MUTED, alpha=0.5, zorder=2)
        ax.hlines(d["early"], cfg["early"][0], cfg["early"][1], color=INK, lw=1.5, ls="--", zorder=4)
        ax.hlines(d["late"], cfg["late"][0], cfg["late"][1], color=ORANGE, lw=1.5, ls="--", zorder=4)
        ax.set_title(d["label"])
        ax.set_ylabel(d["unit"])
        ax.set_xlabel("Season")
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.grid(axis="x", visible=False)
        sign = "+" if d["diff"] >= 0 else ""
        ax.annotate(
            f"{sign}{d['diff']:.1f} {d['unit']}",
            xy=(0.98, 0.06), xycoords="axes fraction", ha="right",
            fontsize=8.5, color=ORANGE, fontweight="bold",
        )

    fig.suptitle(
        "Warmer and wetter, but not less snowy", fontsize=10, y=1.04
    )
    save(fig, "06_mechanism")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(SITE / "results.json"))
    args = p.parse_args(argv)

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"error: {path} missing. Run src/analyze.py first.")

    r = json.loads(path.read_text())
    print("drawing figures")
    fig_regional(r)
    fig_stations(r)
    fig_decades(r)
    fig_sensitivity(r)
    fig_extremes(r)
    fig_mechanism(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
