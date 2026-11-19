#!/usr/bin/env python3
"""Analyse the per-station, per-season snowfall table and emit results.json.

The whole report is driven by this file. Every number that appears on the web
page comes from the JSON this script writes, so the page can never drift out of
sync with the analysis.

Two different comparisons are made, and the report explains why they disagree:

1. Paired stations. Keep stations with enough usable seasons in BOTH periods,
   and compare each station's own early and late mean. Resampling *stations*
   answers "would a different network show the same thing?".

2. Regional anomaly index. Convert each season to an anomaly against that
   station's own 1941-2000 baseline, then average across whatever stations
   reported that season. Resampling *seasons* answers "would a different twenty
   years have shown the same thing?" - which is much closer to the question the
   uncle actually asked.

The second is the headline. The first is supporting evidence.

Usage
-----
    python src/analyze.py
    python src/analyze.py --coverage 0.90 --window jul-jun
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"
SITE = ROOT / "build"

EARLY = (1941, 2000)
LATE = (2001, 2020)

RNG = np.random.default_rng(20261119)


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy, so the reader can follow every line)
# ---------------------------------------------------------------------------
def sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median of all pairwise slopes. Robust to outliers, unlike OLS."""
    i, j = np.triu_indices(len(x), k=1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i])))


def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall trend test. Returns (S statistic, two-sided p-value)."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += int(np.sign(x[i + 1 :] - x[i]).sum())
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return float(s), float(p)


def lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x[:-1] * x[1:]).sum() / denom) if denom else 0.0


def block_resample(x: np.ndarray, block: int) -> np.ndarray:
    """Moving-block bootstrap sample, preserving short-range autocorrelation."""
    n = len(x)
    if n <= block:
        return RNG.choice(x, n, replace=True)
    starts = RNG.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    out = np.concatenate([x[s : s + block] for s in starts])
    return out[:n]


def bootstrap_ci(x: np.ndarray, statistic, *, block: int = 3, n_boot: int = 5000) -> list[float]:
    vals = np.array([statistic(block_resample(x, block)) for _ in range(n_boot)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def difference_ci(early: np.ndarray, late: np.ndarray, *, block: int = 3, n_boot: int = 5000) -> list[float]:
    """CI for mean(late) - mean(early), resampling whole seasons in each period."""
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = block_resample(late, block).mean() - block_resample(early, block).mean()
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def period_mean_ci(x: np.ndarray, *, block: int = 3, n_boot: int = 5000) -> list[float]:
    return bootstrap_ci(x, np.mean, block=block, n_boot=n_boot)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(window: str, coverage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all windows, usable rows of the primary window).

    The full frame is kept so the sensitivity section can sweep every season
    window without re-reading the file.
    """
    path = DERIVED / "station_seasons.csv"
    if not path.exists():
        raise SystemExit(f"error: {path} missing. Run src/parse_ghcn.py first.")

    df = pd.read_csv(path)
    df = df[df["season"].between(EARLY[0], LATE[1])].copy()

    primary = df[df["window"] == window].copy()
    primary["usable"] = (primary["snow_coverage"] >= coverage) & primary["snow_total_in"].notna()
    return df, primary[primary["usable"]].copy()


# ---------------------------------------------------------------------------
# 1. Paired stations
# ---------------------------------------------------------------------------
def paired_analysis(usable: pd.DataFrame, min_seasons: int) -> dict:
    rows = []
    for (sid, name, state, lat, lon, elev), g in usable.groupby(
        ["id", "name", "state", "lat", "lon", "elev"]
    ):
        e = g[g["season"].between(*EARLY)]["snow_total_in"]
        l = g[g["season"].between(*LATE)]["snow_total_in"]
        if len(e) < min_seasons or len(l) < min_seasons:
            continue
        rows.append(
            {
                "id": sid,
                "name": name,
                "state": state,
                "lat": float(lat),
                "lon": float(lon),
                "elev_m": float(elev),
                "early_mean": float(e.mean()),
                "late_mean": float(l.mean()),
                "change": float(l.mean() - e.mean()),
                "n_early": int(len(e)),
                "n_late": int(len(l)),
            }
        )

    if len(rows) < 5:
        raise SystemExit("error: fewer than 5 paired stations; relax --min-seasons")

    stations = pd.DataFrame(rows).sort_values("change").reset_index(drop=True)
    change = stations["change"].to_numpy()

    station_boot = np.array(
        [RNG.choice(change, len(change), replace=True).mean() for _ in range(5000)]
    )

    return {
        "n": int(len(change)),
        "mean_change": float(change.mean()),
        "median_change": float(np.median(change)),
        "sd_change": float(change.std(ddof=1)),
        "ci": [
            float(np.percentile(station_boot, 2.5)),
            float(np.percentile(station_boot, 97.5)),
        ],
        "n_down": int((change < 0).sum()),
        "n_up": int((change > 0).sum()),
        "frac_down": float((change < 0).mean()),
        "min_seasons_required": int(min_seasons),
        "stations": stations.round(2).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 2. Regional anomaly index
# ---------------------------------------------------------------------------
def regional_analysis(usable: pd.DataFrame, min_baseline: int, min_stations: int) -> dict:
    base = (
        usable[usable["season"].between(*EARLY)]
        .groupby("id")["snow_total_in"]
        .agg(["mean", "count"])
    )
    base = base[base["count"] >= min_baseline]
    if len(base) < 5:
        raise SystemExit("error: too few stations with a usable baseline")

    panel = usable[usable["id"].isin(base.index)].copy()
    panel["anomaly"] = panel["snow_total_in"] - panel["id"].map(base["mean"])

    regional = (
        panel.groupby("season")
        .agg(anomaly=("anomaly", "mean"), n=("anomaly", "size"))
        .query("n >= @min_stations")
    )

    series = regional["anomaly"]
    early = series.loc[EARLY[0] : EARLY[1]].to_numpy()
    late = series.loc[LATE[0] : LATE[1]].to_numpy()

    baseline_level = float(base["mean"].mean())
    diff = float(late.mean() - early.mean())

    x = series.index.to_numpy(dtype=float)
    y = series.to_numpy()
    slope = sen_slope(x, y)
    _, p_value = mann_kendall(y)

    decades = []
    for start in range(1940, 2021, 10):
        block = series.loc[start : min(start + 9, 2020)]
        if len(block) < 8:  # a decade needs most of its seasons to mean anything
            continue
        decades.append(
            {
                "decade": f"{start}s",
                "mean": float(block.mean()),
                "se": float(block.std(ddof=1) / math.sqrt(len(block))),
                "n": int(len(block)),
            }
        )

    return {
        "n_stations": int(len(base)),
        "baseline_level_in": baseline_level,
        "early_mean_anomaly": float(early.mean()),
        "late_mean_anomaly": float(late.mean()),
        "early_level_in": baseline_level + float(early.mean()),
        "late_level_in": baseline_level + float(late.mean()),
        "diff": diff,
        "ci": difference_ci(early, late),
        "early_se": float(early.std(ddof=1) / math.sqrt(len(early))),
        "late_se": float(late.std(ddof=1) / math.sqrt(len(late))),
        "early_sd": float(early.std(ddof=1)),
        "late_sd": float(late.std(ddof=1)),
        "sen_slope_per_decade": slope * 10,
        "mk_p": p_value,
        "lag1_autocorr": lag1_autocorr(y),
        "min_stations_per_season": int(regional["n"].min()),
        "max_stations_per_season": int(regional["n"].max()),
        "decades": decades,
        "series": [
            {"season": int(s), "anomaly": float(a), "n": int(n)}
            for s, a, n in zip(regional.index, regional["anomaly"], regional["n"])
        ],
        "station_changes_available": int(len(base)),
    }


# ---------------------------------------------------------------------------
# 3. Extremes and persistence
# ---------------------------------------------------------------------------
def extremes_analysis(usable: pd.DataFrame, threshold_coverage: float = 0.95) -> dict:
    """Counts of heavy-snow days and snow-on-ground days, on near-complete seasons only.

    Restricting to >95% coverage matters here: a season missing 20% of its days
    would miss roughly 20% of its heavy-snow days, which would look like a trend.
    """
    d = usable[usable["snow_coverage"] >= threshold_coverage]
    out = {"coverage_threshold": threshold_coverage}

    for column, label in [
        ("snow_days_ge6", "days_ge6"),
        ("snow_days_ge12", "days_ge12"),
        ("snwd_days_ge1", "days_with_snow_cover"),
    ]:
        e = d[d["season"].between(*EARLY)].groupby("season")[column].mean()
        l = d[d["season"].between(*LATE)].groupby("season")[column].mean()
        if len(e) < 10 or len(l) < 10:
            out[label] = None
            continue
        both = pd.concat([e, l]).sort_index()
        out[label] = {
            "early": float(e.mean()),
            "late": float(l.mean()),
            "diff": float(l.mean() - e.mean()),
            "ci": difference_ci(e.to_numpy(), l.to_numpy()),
            "n_early_seasons": int(len(e)),
            "n_late_seasons": int(len(l)),
            "series": [
                {"season": int(s), "value": float(v)} for s, v in both.items()
            ],
        }

    return out


# ---------------------------------------------------------------------------
# 4. Mechanism: precipitation and temperature
# ---------------------------------------------------------------------------
def mechanism_analysis(usable: pd.DataFrame) -> dict:
    out = {}
    for column, key, label, unit in [
        ("prcp_total_in", "precip", "Seasonal precipitation", "in"),
        ("tmean_c", "temp", "Seasonal mean temperature", "\u00b0C"),
    ]:
        d = usable[usable[column].notna()]
        e = d[d["season"].between(*EARLY)].groupby("season")[column].mean()
        l = d[d["season"].between(*LATE)].groupby("season")[column].mean()
        both = pd.concat([e, l]).sort_index()
        out[key] = {
            "label": label,
            "unit": unit,
            "early": float(e.mean()),
            "late": float(l.mean()),
            "diff": float(l.mean() - e.mean()),
            "ci": difference_ci(e.to_numpy(), l.to_numpy()),
            "series": [
                {"season": int(s), "value": float(v)} for s, v in both.items()
            ],
        }
    return out


# ---------------------------------------------------------------------------
# 4b. Artefacts: is the "snowy places lost most" pattern real?
# ---------------------------------------------------------------------------
def artefact_check(usable: pd.DataFrame, paired: dict) -> dict:
    """Test whether the elevation/change correlation is regression to the mean.

    Stations that started with a high mean can only fall, and a noisy 20-season
    window makes that fall look systematic. If the elevation correlation
    collapses once the early-period mean is held constant, it is an artefact.
    """
    stations = pd.DataFrame(paired["stations"])
    change = stations["change"].to_numpy()
    elev = stations["elev_m"].to_numpy()
    early = stations["early_mean"].to_numpy()

    def residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return y - np.polyval(np.polyfit(x, y, 1), x)

    e = usable[usable["season"].between(*EARLY)]
    first_half = e[e["season"] <= 1970].groupby("id")["snow_total_in"].mean()
    second_half = e[e["season"] >= 1971].groupby("id")["snow_total_in"].mean()
    both = pd.concat([first_half, second_half], axis=1, keys=["a", "b"]).dropna()
    reliability = float(np.corrcoef(both["a"], both["b"])[0, 1]) if len(both) > 3 else float("nan")

    return {
        "corr_change_elevation": float(np.corrcoef(change, elev)[0, 1]),
        "corr_change_early_mean": float(np.corrcoef(change, early)[0, 1]),
        "partial_corr_change_elevation_given_early_mean": float(
            np.corrcoef(residuals(change, early), residuals(elev, early))[0, 1]
        ),
        "baseline_split_half_corr": reliability,
        "n_stations": int(len(stations)),
    }


# ---------------------------------------------------------------------------
# 5. Sensitivity
# ---------------------------------------------------------------------------
def sensitivity(df: pd.DataFrame, windows: list[str], coverages: list[float], min_baseline: int, min_stations: int) -> list[dict]:
    rows = []
    for window in windows:
        for coverage in coverages:
            sub = df[(df["window"] == window) & (df["snow_coverage"] >= coverage) & df["snow_total_in"].notna()]
            if sub.empty:
                continue
            try:
                r = regional_analysis(sub, min_baseline, min_stations)
            except SystemExit:
                continue
            rows.append(
                {
                    "window": window,
                    "coverage": coverage,
                    "n_stations": r["n_stations"],
                    "diff": r["diff"],
                    "ci": r["ci"],
                    "sen_slope_per_decade": r["sen_slope_per_decade"],
                    "mk_p": r["mk_p"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
def decide(regional: dict, paired: dict) -> dict:
    """The rule is fixed in advance; this function only reports which legs pass.

    Yes requires all four. Anything else is No, with the failing leg named.
    """
    checks = [
        {
            "id": "direction",
            "label": "Regional mean is lower in 2001-2020",
            "passed": regional["diff"] < 0,
            "detail": f"{regional['diff']:+.1f} in",
        },
        {
            "id": "uncertainty",
            "label": "95% CI on that difference excludes zero",
            "passed": regional["ci"][0] > 0 or regional["ci"][1] < 0,
            "detail": f"[{regional['ci'][0]:+.1f}, {regional['ci'][1]:+.1f}] in",
        },
        {
            "id": "stations",
            "label": "Median station change negative and >=60% of stations declining",
            "passed": paired["median_change"] < 0 and paired["frac_down"] >= 0.60,
            "detail": f"median {paired['median_change']:+.1f} in, {paired['frac_down']*100:.0f}% down",
        },
        {
            "id": "trend",
            "label": "Whole-record trend negative at p < 0.05",
            "passed": regional["sen_slope_per_decade"] < 0 and regional["mk_p"] < 0.05,
            "detail": f"{regional['sen_slope_per_decade']:+.2f} in/decade, p={regional['mk_p']:.3f}",
        },
    ]
    return {
        "verdict": "Yes" if all(c["passed"] for c in checks) else "No",
        "checks": checks,
        "n_passed": sum(c["passed"] for c in checks),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coverage", type=float, default=0.80, help="minimum season coverage (default 0.80)")
    p.add_argument("--window", default="jul-jun", help="primary season window (default jul-jun)")
    p.add_argument("--min-seasons", type=int, default=12, help="min usable seasons per period per station")
    p.add_argument("--min-baseline", type=int, default=30, help="min baseline seasons for the regional index")
    p.add_argument("--min-stations", type=int, default=5, help="min stations reporting in a given season")
    args = p.parse_args(argv)

    df, usable = load(args.window, args.coverage)
    if usable.empty:
        print("error: no usable rows; coverage threshold too strict", file=sys.stderr)
        return 1

    print(f"window={args.window}  coverage>={args.coverage}  usable station-seasons={len(usable):,}")

    paired = paired_analysis(usable, args.min_seasons)
    print(f"paired stations: {paired['n']}  change {paired['mean_change']:+.1f} in")

    regional = regional_analysis(usable, args.min_baseline, args.min_stations)
    print(f"regional index : {regional['diff']:+.1f} in  CI [{regional['ci'][0]:+.1f}, {regional['ci'][1]:+.1f}]")

    extremes = extremes_analysis(usable)
    mechanism = mechanism_analysis(usable)
    artefacts = artefact_check(usable, paired)

    sensitivity_rows = sensitivity(
        df,
        windows=["jul-jun", "oct-may", "nov-mar"],
        coverages=[0.50, 0.70, 0.80, 0.90, 0.95],
        min_baseline=args.min_baseline,
        min_stations=args.min_stations,
    )
    print(f"sensitivity    : {len(sensitivity_rows)} combinations")

    decision = decide(regional, paired)
    print(f"verdict        : {decision['verdict']} ({decision['n_passed']}/4 criteria passed)")

    meta = pd.read_csv(DERIVED / "station_seasons.csv", usecols=["id", "name", "state"])
    meta = meta.drop_duplicates()

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "window": args.window,
            "coverage": args.coverage,
            "early": list(EARLY),
            "late": list(LATE),
            "min_seasons": args.min_seasons,
            "min_baseline": args.min_baseline,
            "min_stations": args.min_stations,
        },
        "dataset": {
            "stations_downloaded": int(meta["id"].nunique()),
            "stations_in_regional_index": regional["n_stations"],
            "stations_paired": paired["n"],
            "station_seasons_usable": int(len(usable)),
            "states": sorted(meta["state"].dropna().unique().tolist()),
        },
        "decision": decision,
        "paired": paired,
        "regional": regional,
        "extremes": extremes,
        "mechanism": mechanism,
        "artefacts": artefacts,
        "sensitivity": sensitivity_rows,
    }

    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"written        : {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
