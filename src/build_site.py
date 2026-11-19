#!/usr/bin/env python3
"""Render build/index.html and build/style.css from results.json and the template.

Every number in the page is injected here from build/results.json, so the prose
cannot drift away from the analysis. Unfilled placeholders are an error, not a
silent blank, because a report with a stray {{TOKEN}} on it is worse than one
that fails the build.

Usage
-----
    python src/build_site.py
    python src/build_site.py --repo-url https://github.com/you/new-england-snow
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "build"
TEMPLATES = ROOT / "templates"
FIGURES = SITE / "figures"

MINUS = "\u2212"  # a real minus sign reads better than a hyphen


def signed(x: float, digits: int = 1) -> str:
    """Format +2.4 / -4.3 with a typographic minus."""
    return f"{x:+.{digits}f}".replace("-", MINUS)


def plain(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace("-", MINUS)


def ci(pair: list[float], digits: int = 1) -> str:
    return f"{signed(pair[0], digits)} to {signed(pair[1], digits)}"


def long_date(stamp: str) -> str:
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return stamp
    return f"{dt.day} {dt.strftime('%B %Y')}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def rules_html(decision: dict) -> tuple[str, int]:
    rows = []
    for check in decision["checks"]:
        cls = "" if check["passed"] else ' class="failed"'
        tag = "met" if check["passed"] else "not met"
        tag_cls = "verdict-tag--met" if check["passed"] else ""
        # Labels are prose, but they are injected into HTML: escape them so a
        # stray "<" or "&" can never break the page.
        label = escape(check["label"])
        detail = escape(check["detail"])
        rows.append(
            f'<li{cls}>{label} '
            f'<span class="verdict-tag {tag_cls}">{tag}</span>'
            f'<span class="rule-detail">{detail}</span></li>'
        )
    return "\n      ".join(rows), decision["n_passed"]


def sensitivity_html(rows: list[dict]) -> str:
    order = {"jul-jun": 0, "oct-may": 1, "nov-mar": 2}
    rows = sorted(rows, key=lambda d: (order[d["window"]], -d["coverage"]))
    out = []
    for r in rows:
        trend = signed(r["sen_slope_per_decade"], 2)
        # Mark the configuration actually used for the headline numbers.
        marker = " class=\"used\"" if (r["window"], round(r["coverage"], 2)) == ("jul-jun", 0.80) else ""
        out.append(
            f"<tr{marker}>"
            f"<td>{r['window']}</td>"
            f'<td class="num">{r["coverage"]:.2f}</td>'
            f'<td class="num">{r["n_stations"]}</td>'
            f'<td class="num {"pos" if r["diff"] >= 0 else "neg"}">{signed(r["diff"])}</td>'
            f'<td class="num">[{ci(r["ci"])}]</td>'
            f'<td class="num">{trend}</td>'
            f'<td class="num">{r["mk_p"]:.3f}</td>'
            f"</tr>"
        )
    return "\n        ".join(out)


def station_html(stations: list[dict]) -> str:
    rows = sorted(stations, key=lambda s: -s["change"])
    out = []
    for s in rows:
        cls = "pos" if s["change"] >= 0 else "neg"
        out.append(
            f"<tr>"
            f"<td>{s['name'].title()}</td>"
            f"<td>{s['state']}</td>"
            f'<td class="num">{s["elev_m"]:.0f}</td>'
            f'<td class="num">{s["early_mean"]:.1f}</td>'
            f'<td class="num">{s["late_mean"]:.1f}</td>'
            f'<td class="num {cls}">{signed(s["change"])}</td>'
            f"</tr>"
        )
    return "\n        ".join(out)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(repo_url: str) -> int:
    results_path = SITE / "results.json"
    if not results_path.exists():
        print(f"error: {results_path} missing. Run src/analyze.py first.", file=sys.stderr)
        return 1

    r = json.loads(results_path.read_text())
    reg, paired, dec = r["regional"], r["paired"], r["decision"]
    ex, mech, art = r["extremes"], r["mechanism"], r["artefacts"]

    figure_names = [
        "01_regional", "02_stations", "03_decades",
        "04_sensitivity", "05_extremes", "06_mechanism",
    ]
    missing = [n for n in figure_names if not (FIGURES / f"{n}.png").exists()]
    if missing:
        print(f"error: missing figures {missing}. Run src/figures.py first.", file=sys.stderr)
        return 1

    rules, n_passed = rules_html(dec)
    sens_p = min(s["mk_p"] for s in r["sensitivity"])

    if n_passed == 0:
        rules_summary = "None of the four criteria were met."
    elif n_passed == len(dec["checks"]):
        rules_summary = "All four criteria were met."
    else:
        rules_summary = f"{n_passed} of the four criteria were met."

    shared = len(set(reg["station_ids"]) & {s["id"] for s in paired["stations"]})

    tokens = {
        # headline
        "VERDICT_WORD": dec["verdict"],
        "VERDICT_TEXT": (
            f"The region's stations averaged <em>{plain(abs(reg['diff']))} inches more</em> snow in "
            f"2001&ndash;2020 than in 1941&ndash;2000. The uncertainty around that number is wider "
            f"than the number itself, so the data neither confirm a decline nor rule a small one out."
        ),
        "N_RULES_PASSED": str(n_passed),
        "RULES_SUMMARY": rules_summary,
        "N_SHARED": str(shared),
        "RULES": rules,
        # stat cards
        "DIFF_SIGNED": signed(reg["diff"]),
        "DIFF_ABS": plain(abs(reg["diff"])),
        "N_UP": str(paired["n_up"]),
        "N_PAIRED": str(paired["n"]),
        "SLOPE": signed(reg["sen_slope_per_decade"], 2),
        "MK_P": f"{reg['mk_p']:.2f}",
        # regional
        "EARLY_LEVEL": plain(reg["early_level_in"]),
        "LATE_LEVEL": plain(reg["late_level_in"]),
        "CI_LOW": signed(reg["ci"][0]),
        "CI_HIGH": signed(reg["ci"][1]),
        "EARLY_SD": plain(reg["early_sd"], 0),
        "LATE_SE": plain(reg["late_se"]),
        "MIN_STATIONS": str(reg["min_stations_per_season"]),
        "MAX_STATIONS": str(reg["max_stations_per_season"]),
        "N_STATIONS_REGIONAL": str(reg["n_stations"]),
        "N_STATIONS_DL": str(r["dataset"]["stations_downloaded"]),
        "RHO": signed(reg["lag1_autocorr"], 2),
        # paired
        "N_DOWN": str(paired["n_down"]),
        "MEDIAN_CHANGE": signed(paired["median_change"]),
        "MEDIAN_ABS": plain(abs(paired["median_change"])),
        # sensitivity
        "N_SENS": str(len(r["sensitivity"])),
        "SENS_MIN_P": f"{sens_p:.2f}",
        "SENS_TABLE": sensitivity_html(r["sensitivity"]),
        # extremes
        "DAYS6_E": plain(ex["days_ge6"]["early"]),
        "DAYS6_L": plain(ex["days_ge6"]["late"]),
        "DAYS12_E": plain(ex["days_ge12"]["early"]),
        "DAYS12_L": plain(ex["days_ge12"]["late"]),
        "COVER_E": plain(ex["days_with_snow_cover"]["early"]),
        "COVER_L": plain(ex["days_with_snow_cover"]["late"]),
        "COVER_DIFF": signed(ex["days_with_snow_cover"]["diff"]),
        "COVER_CI": ci(ex["days_with_snow_cover"]["ci"]),
        # mechanism
        "PRECIP_E": plain(mech["precip"]["early"]),
        "PRECIP_L": plain(mech["precip"]["late"]),
        "PRECIP_DIFF": signed(mech["precip"]["diff"]),
        "TEMP_E": plain(mech["temp"]["early"], 2),
        "TEMP_L": plain(mech["temp"]["late"], 2),
        # artefact
        "CORR_ELEV": signed(art["corr_change_elevation"], 2),
        "PARTIAL": signed(art["partial_corr_change_elevation_given_early_mean"], 2),
        # meta
        "SNAPSHOT": long_date(r["dataset"]["downloaded_utc"]),
        "COVERAGE_PCT": f"{r['config']['coverage'] * 100:.0f}%",
        "GENERATED": long_date(r["generated_utc"]),
        "REPO_URL": repo_url,
        "STATION_TABLE": station_html(paired["stations"]),
    }

    template = (TEMPLATES / "report.html").read_text()
    html = template
    for key, value in tokens.items():
        html = html.replace("{{" + key + "}}", str(value))

    leftover = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", html)))
    if leftover:
        print(f"error: unfilled template tokens: {', '.join(leftover)}", file=sys.stderr)
        return 1

    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "index.html").write_text(html)
    shutil.copyfile(TEMPLATES / "style.css", SITE / "style.css")

    # GitHub Pages serves 404.html for unknown paths.
    notfound = (
        '<!doctype html><meta charset="utf-8"><title>Not found</title>'
        '<link rel="stylesheet" href="/style.css">'
        '<main class="wrap measure" style="padding:6rem 0">'
        "<h1>Not found</h1><p>That page does not exist. "
        '<a href="/">Return to the report</a>.</p></main>'
    )
    (SITE / "404.html").write_text(notfound)
    (SITE / ".nojekyll").write_text("")

    print(f"wrote build/index.html ({len(html) / 1024:.0f} KB)")
    print("wrote build/style.css, build/404.html, build/.nojekyll")
    print(f"verdict: {dec['verdict']}  ({n_passed}/4 criteria met)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-url", default=None, help="repository URL shown in the footer")
    args = p.parse_args(argv)

    repo_url = args.repo_url
    if repo_url is None:
        config = ROOT / "site_config.json"
        if config.exists():
            repo_url = json.loads(config.read_text()).get("repo_url", "")
        else:
            repo_url = ""
    if not repo_url:
        print("warning: no --repo-url and no site_config.json; the footer link will be blank", file=sys.stderr)

    return build(repo_url)


if __name__ == "__main__":
    raise SystemExit(main())
