#!/usr/bin/env python3
"""Parse GHCN-Daily .dly files into per-station, per-season aggregates.

Why aggregates instead of a daily table? The daily record for 67 stations over
80 years is ~2 million rows and ~150 MB. We only ever ask it questions of the
form "how much snow fell this winter at this station", so we answer them once,
here, and emit a table of roughly 20,000 rows.

Season definitions
------------------
Each season is labelled by the calendar year containing its January, so that
season 2001 means "the winter of 2000/01".

    jul-jun : 1 Jul (Y-1) .. 30 Jun (Y)     primary definition
    oct-may : 1 Oct (Y-1) .. 31 May (Y)     sensitivity check
    nov-mar : 1 Nov (Y-1) .. 31 Mar (Y)     sensitivity check

The jul-jun definition keeps a single storm inside a single season and still
catches October and late-April snow. The two other windows exist only so the
report can show the answer does not depend on this choice.

Units
-----
In GHCN-Daily .dly files:
    SNOW, SNWD  millimetres
    PRCP        tenths of a millimetre
    TMAX, TMIN  tenths of a degree Celsius
    -9999       missing
Values carrying a non-blank QFLAG have failed a GHCN quality check and are
treated as missing here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DLY = RAW / "dly"
DERIVED = ROOT / "data" / "derived"

ELEMENTS = ("SNOW", "SNWD", "PRCP", "TMAX", "TMIN")
MM_PER_INCH = 25.4
TENTHS_MM_PER_INCH = 254.0

# name -> (first month of season, last month of season)
WINDOWS = {
    "jul-jun": (7, 6),
    "oct-may": (10, 5),
    "nov-mar": (11, 3),
}

VALUE_SPECS = [(21 + 8 * d, 26 + 8 * d) for d in range(31)]
QFLAG_SPECS = [(27 + 8 * d, 28 + 8 * d) for d in range(31)]
COLSPECS = [(0, 11), (11, 15), (15, 17), (17, 21)] + VALUE_SPECS + QFLAG_SPECS
NAMES = (
    ["id", "year", "month", "element"]
    + [f"v{d}" for d in range(31)]
    + [f"q{d}" for d in range(31)]
)


def read_dly(path: Path) -> pd.DataFrame:
    """Read one .dly file into a long frame of (date, element, value) rows."""
    raw = pd.read_fwf(path, colspecs=COLSPECS, names=NAMES, dtype=str, header=None)
    raw = raw[raw["element"].isin(ELEMENTS)]
    if raw.empty:
        return pd.DataFrame(columns=["id", "date", "element", "value"])

    n = len(raw)
    long = pd.DataFrame(
        {
            "id": np.repeat(raw["id"].to_numpy(), 31),
            "year": np.repeat(raw["year"].to_numpy(), 31),
            "month": np.repeat(raw["month"].to_numpy(), 31),
            "element": np.repeat(raw["element"].to_numpy(), 31),
            "day": np.tile(np.arange(1, 32), n),
            "value": raw[[f"v{d}" for d in range(31)]].to_numpy().ravel(),
            "qflag": raw[[f"q{d}" for d in range(31)]].to_numpy().ravel(),
        }
    )

    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long.loc[long["value"] == -9999, "value"] = np.nan
    long["qflag"] = long["qflag"].fillna(" ").str.strip()
    long.loc[long["qflag"] != "", "value"] = np.nan

    long["date"] = pd.to_datetime(
        {"year": long["year"], "month": long["month"], "day": long["day"]},
        errors="coerce",
    )
    # Day 30/31 in short months parses to NaT and is dropped here.
    long = long.dropna(subset=["date"])
    return long[["id", "date", "element", "value"]]


def daily_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot one station's long frame to (date x element), converting units."""
    wide = (
        long.pivot_table(index="date", columns="element", values="value", aggfunc="first")
        .reindex(columns=list(ELEMENTS))
        .sort_index()
    )
    out = pd.DataFrame(index=wide.index)
    out["snow_in"] = wide["SNOW"] / MM_PER_INCH
    out["snwd_in"] = wide["SNWD"] / MM_PER_INCH
    out["prcp_in"] = wide["PRCP"] / TENTHS_MM_PER_INCH
    out["tmax_c"] = wide["TMAX"] / 10.0
    out["tmin_c"] = wide["TMIN"] / 10.0
    out["tmean_c"] = (out["tmax_c"] + out["tmin_c"]) / 2.0
    return out


def season_summary(daily: pd.DataFrame, station_id: str, window: str) -> pd.DataFrame:
    """Aggregate a station's daily record into one row per season for one window."""
    start_month, end_month = WINDOWS[window]

    dates = daily.index
    label = dates.year + (dates.month >= start_month)
    in_window = (dates.month >= start_month) | (dates.month <= end_month)
    d = daily[in_window]
    season = label[in_window]

    snow = d["snow_in"]
    by_snow = snow.groupby(season)
    dates_by_season = d.index.to_series().groupby(season)

    summary = pd.DataFrame(
        {
            "snow_total_in": by_snow.sum(min_count=1),
            "snow_days_obs": by_snow.count(),
            "snow_days_ge1": (snow >= 1).groupby(season).sum(),
            "snow_days_ge6": (snow >= 6).groupby(season).sum(),
            "snow_days_ge12": (snow >= 12).groupby(season).sum(),
            "snwd_max_in": d["snwd_in"].groupby(season).max(),
            "snwd_days_ge1": (d["snwd_in"] >= 1).groupby(season).sum(),
            "prcp_total_in": d["prcp_in"].groupby(season).sum(min_count=1),
            "prcp_days_obs": d["prcp_in"].groupby(season).count(),
            "tmean_c": d["tmean_c"].groupby(season).mean(),
            "tmean_days_obs": d["tmean_c"].groupby(season).count(),
            "first_valid": dates_by_season.min(),
            "last_valid": dates_by_season.max(),
        }
    )

    # Denominator: days actually inside the season window, not days observed.
    # A season at the very edge of a record scores low and is filtered later.
    idx = summary.index.to_numpy()
    starts = pd.to_datetime([f"{y - 1}-{start_month:02d}-01" for y in idx])
    ends = pd.to_datetime([f"{y}-{end_month:02d}-01" for y in idx]) + pd.offsets.MonthEnd(0)
    summary["season_days"] = (ends - starts).days + 1
    summary["snow_coverage"] = summary["snow_days_obs"] / summary["season_days"]

    summary.insert(0, "id", station_id)
    summary.insert(1, "window", window)
    summary.index.name = "season"
    return summary.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="parse only the first N stations")
    args = parser.parse_args(argv)

    if not DLY.exists() or not any(DLY.glob("*.dly")):
        print(f"error: no .dly files in {DLY}. Run src/fetch_data.py first.", file=sys.stderr)
        return 1

    DERIVED.mkdir(parents=True, exist_ok=True)

    files = sorted(DLY.glob("*.dly"))
    if args.limit:
        files = files[: args.limit]

    frames = []
    for i, path in enumerate(files, 1):
        long = read_dly(path)
        if long.empty:
            print(f"  warning: {path.name} has no usable elements", file=sys.stderr)
            continue
        daily = daily_wide(long)
        for window in WINDOWS:
            frames.append(season_summary(daily, path.stem, window))
        if i % 10 == 0 or i == len(files):
            print(f"  parsed {i}/{len(files)} stations")

    if not frames:
        print("error: nothing parsed", file=sys.stderr)
        return 1

    seasons = pd.concat(frames, ignore_index=True)

    meta = pd.read_csv(RAW / "stations_selected.csv")
    seasons = seasons.merge(
        meta[["id", "name", "state", "lat", "lon", "elev"]], on="id", how="left"
    )

    out = DERIVED / "station_seasons.csv"
    seasons.to_csv(out, index=False)

    print()
    print(f"stations parsed : {seasons['id'].nunique()}")
    print(f"rows            : {len(seasons):,} across {len(WINDOWS)} season windows")
    print(f"seasons covered : {seasons['season'].min()}-{seasons['season'].max()}")
    print(f"written         : {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")

    # A quick look at the thing we actually care about, so a bad parse is obvious.
    strong = seasons[(seasons["window"] == "jul-jun") & (seasons["snow_coverage"] >= 0.90)]
    early = strong[strong["season"].between(1941, 2000)]["snow_total_in"]
    late = strong[strong["season"].between(2001, 2020)]["snow_total_in"]
    print()
    print("sanity check (jul-jun, coverage >= 90%, unweighted station-seasons):")
    print(f"  1941-2000  n={len(early):>5,}  mean={early.mean():6.1f} in")
    print(f"  2001-2020  n={len(late):>5,}  mean={late.mean():6.1f} in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
