#!/usr/bin/env python3
"""Download GHCN-Daily station metadata and snowfall records for New England.

GHCN-Daily is the daily-resolution member of the Global Historical Climatology
Network. It is the only public GHCN product that carries daily snowfall (SNOW)
and snow depth (SNWD), which is what this project needs.

The script is deliberately boring: fetch three small metadata files, filter the
inventory down to New England stations with snowfall records spanning both
study periods, then fetch one .dly file per station.

Everything lands in data/, which .gitignore excludes. Nothing here is committed.

Usage
-----
    python src/fetch_data.py                # full download
    python src/fetch_data.py --check        # dry run, print selection only
    python src/fetch_data.py --max-stations 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DLY = RAW / "dly"

# The six New England states, as GHCN state codes.
NEW_ENGLAND = {"CT", "RI", "MA", "VT", "NH", "ME"}

# Station must have snowfall records reaching back to at least this year...
MIN_FIRST_YEAR = 1945
# ...and forward to at least this one, so it can serve both study periods.
MIN_LAST_YEAR = 2018

USER_AGENT = "new-england-snow/1.0 (student coursework)"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def download(url: str, dest: Path, *, retries: int = 4, force: bool = False) -> Path:
    """Download url to dest, skipping if already present. Retries with backoff."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                expected = resp.headers.get("Content-Length")
                with open(tmp, "wb") as fh:
                    while chunk := resp.read(1 << 16):
                        fh.write(chunk)
            # Guard against a proxy returning a short or empty body.
            if expected and tmp.stat().st_size != int(expected):
                raise IOError(f"short read: {tmp.stat().st_size} != {expected} bytes")
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"failed to fetch {url}: {exc}") from exc
            time.sleep(2 ** attempt)  # 2, 4, 8s

    return dest


# ---------------------------------------------------------------------------
# Fixed-width metadata parsing
# ---------------------------------------------------------------------------
def read_stations(path: Path) -> dict[str, dict]:
    """Parse ghcnd-stations.txt into {station_id: metadata}.

    Columns (1-based, from the GHCN readme):
        ID 1-11, LAT 13-20, LON 22-30, ELEV 32-37, STATE 39-40,
        NAME 42-71, GSN 73-75, HCN/CRN 77-79, WMO 81-85
    """
    stations: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 40:
                continue
            sid = line[0:11].strip()
            if not sid:
                continue
            stations[sid] = {
                "id": sid,
                "lat": _to_float(line[12:20]),
                "lon": _to_float(line[21:30]),
                "elev": _to_float(line[31:37]),
                "state": line[38:40].strip(),
                "name": line[41:71].strip(),
            }
    return stations


def read_inventory(path: Path) -> list[dict]:
    """Parse ghcnd-inventory.txt.

    Columns: ID 1-11, LAT 13-20, LON 22-30, ELEMENT 32-35,
             FIRSTYEAR 37-40, LASTYEAR 42-45
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 45:
                continue
            sid = line[0:11].strip()
            if not sid:
                continue
            rows.append(
                {
                    "id": sid,
                    "element": line[31:35].strip(),
                    "first_year": _to_int(line[36:40]),
                    "last_year": _to_int(line[41:45]),
                }
            )
    return rows


def _to_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def _to_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_stations(
    stations: dict[str, dict],
    inventory: list[dict],
    *,
    min_first: int = MIN_FIRST_YEAR,
    min_last: int = MIN_LAST_YEAR,
) -> list[dict]:
    """Keep New England stations whose SNOW record spans both study periods."""
    snow_span = {
        row["id"]: (row["first_year"], row["last_year"])
        for row in inventory
        if row["element"] == "SNOW"
    }

    selected = []
    for sid, meta in stations.items():
        if meta["state"] not in NEW_ENGLAND:
            continue
        span = snow_span.get(sid)
        if span is None:
            continue
        first, last = span
        if first is None or last is None:
            continue
        if first > min_first or last < min_last:
            continue
        selected.append({**meta, "snow_first": first, "snow_last": last})

    selected.sort(key=lambda s: (s["state"], s["name"]))
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="print selection, download nothing")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--max-stations", type=int, default=None, help="cap station count (smoke test)")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (default 8)")
    args = parser.parse_args(argv)

    RAW.mkdir(parents=True, exist_ok=True)

    print("[1/3] fetching station metadata")
    stations_path = download(f"{BASE_URL}/ghcnd-stations.txt", RAW / "ghcnd-stations.txt", force=args.force)
    inventory_path = download(f"{BASE_URL}/ghcnd-inventory.txt", RAW / "ghcnd-inventory.txt", force=args.force)
    # Keep the readme next to the data: it is the authority on units and flags.
    download(f"{BASE_URL}/readme.txt", RAW / "ghcnd-readme.txt", force=args.force)

    print("[2/3] selecting stations")
    stations = read_stations(stations_path)
    inventory = read_inventory(inventory_path)
    selected = select_stations(stations, inventory)
    print(f"      {len(stations):,} stations in file, {len(selected)} New England stations qualify")

    if args.max_stations is not None:
        selected = selected[: args.max_stations]

    if args.check:
        print(f"{'ID':<12} {'ST':<3} {'SNOW years':<12} NAME")
        for s in selected:
            print(f"{s['id']:<12} {s['state']:<3} {s['snow_first']}-{s['snow_last']:<7} {s['name']}")
        return 0

    if not selected:
        print("error: selection is empty; check the metadata formats", file=sys.stderr)
        return 1

    print(f"[3/3] downloading {len(selected)} station files into {DLY}")
    DLY.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download,
                f"{BASE_URL}/all/{s['id']}.dly",
                DLY / f"{s['id']}.dly",
                force=args.force,
            ): s["id"]
            for s in selected
        }
        done = 0
        for fut in as_completed(futures):
            sid = futures[fut]
            done += 1
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((sid, str(exc)))
            if done % 25 == 0 or done == len(futures):
                print(f"      {done}/{len(futures)}")

    # Manifest: the station list is small and useful to inspect, but still ignored.
    manifest = RAW / "stations_selected.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as fh:
        cols = ["id", "state", "name", "lat", "lon", "elev", "snow_first", "snow_last", "bytes"]
        fh.write(",".join(cols) + "\n")
        for s in selected:
            path = DLY / f"{s['id']}.dly"
            size = path.stat().st_size if path.exists() else 0
            name = s["name"].replace(",", ";")
            fh.write(
                f"{s['id']},{s['state']},{name},{s['lat']},{s['lon']},"
                f"{s['elev']},{s['snow_first']},{s['snow_last']},{size}\n"
            )

    total_mb = sum(p.stat().st_size for p in DLY.glob("*.dly")) / 1e6
    print(f"      data/raw/dly: {total_mb:,.1f} MB across {len(list(DLY.glob('*.dly')))} files")
    print(f"      manifest: {manifest.relative_to(ROOT)}")

    # Record the snapshot: GHCN-Daily is updated daily, so the report must be
    # able to say which vintage of the data it describes.
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RAW / "download_stamp.txt").write_text(stamp + "\n")
    print(f"      snapshot: {stamp}")

    if failures:
        print(f"\n{len(failures)} station(s) failed:", file=sys.stderr)
        for sid, msg in failures:
            print(f"  {sid}: {msg}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
