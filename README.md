# Was there less snow in New England?

**Report: <https://mwm1070.github.io/new-england-snow/>**

## The answer: no

New England seasonal snowfall did not decline between 1941–2000 and 2001–2020.
Across 59 stations with usable records in both periods, the region averaged
**2.4 inches more** snow in the recent period, not less.

The more useful finding is about uncertainty. Year-to-year snowfall varies by
about 17 inches, so a 20-season average carries a standard error near 4 inches
and the 95% confidence interval on the change runs from **−4.3 to +8.4 inches**.
The data therefore cannot confirm a decline, and they also cannot rule out a
small one. A two-decade window is not enough evidence to support the claim in
either direction — which is the honest conclusion, and a different one from
"he was wrong".

The full reasoning, six figures and the robustness checks are on the report page.

## What's here

```
src/fetch_data.py     download GHCN-Daily station metadata and 67 .dly files
src/parse_ghcn.py     fixed-width parser -> per-station, per-season aggregates
src/analyze.py        statistics, decision rule, sensitivity -> build/results.json
src/figures.py        six figures from results.json
src/build_site.py     render build/index.html from the template
templates/            report.html (the prose) and style.css
.github/workflows/    builds and publishes the site on every push
```

`data/` and `build/` are git-ignored and do not exist in this repository or in
its history. The raw data is re-downloaded by the workflow on every build.

## Reproducing it

```bash
python3 -m pip install -r requirements.txt
python3 src/fetch_data.py     # ~135 MB into data/raw (git-ignored)
python3 src/parse_ghcn.py
python3 src/analyze.py
python3 src/figures.py
python3 src/build_site.py
open build/index.html
```

The analysis takes about a minute end to end, most of it spent parsing the
fixed-width files.

## Data

[GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
from NOAA's National Centers for Environmental Information. It is a US
Government work in the public domain.

Stations are selected from Connecticut, Rhode Island, Massachusetts, Vermont,
New Hampshire and Maine, and must have snowfall records spanning both study
periods. Values of `-9999` are treated as missing, as are values carrying a
non-blank GHCN quality flag. Snowfall and snow depth are stored in millimetres
in the source files and converted to inches.

## Method in brief

- **Seasons** run 1 July to 30 June, labelled by the year containing January, so
  season 2001 is the winter of 2000/01. This keeps a storm inside one season and
  still catches October and late-April snow.
- **A usable season** needs at least 80% of its days reporting snowfall. Missing
  days are never counted as zero.
- **Two comparisons.** Paired stations compares each station against itself using
  only stations with at least 12 usable seasons in each period. The regional index
  converts every season to an anomaly against that station's own 1941–2000 mean and
  averages across stations, which uses far more of the record and is the headline.
- **Uncertainty** from a moving-block bootstrap (3-season blocks, 5,000 resamples),
  which respects the mild year-to-year correlation in the series.
- **Trends** from Sen's slope and the Mann–Kendall test, both rank-based.
- **The verdict rule was fixed before looking at the data**, and the page reports
  all four criteria and which ones passed.

Six figures, the full station table, the sensitivity matrix and the limitations
are on the report page.
