# Was there less snow in New England?

**Report: <https://mwm1070.github.io/new-england-snow/>**

## The answer: no

New England seasonal snowfall did not decline between 1941–2000 and 2001–2020.
Across 59 stations with usable records in both periods, the region averaged
**2.4 inches more** snow in the recent period, not less.

The more useful finding is about uncertainty. Year-to-year snowfall varies by
about 17 inches, so a 20-season average carries a standard error near 4 inches
and the 95% confidence interval on the change runs from **−4.3 to +8.4 inches**.
The data therefore cannot confirm a decline. A two-decade window is not enough evidence to support the claim in
either direction however — which is the honest conclusion, and a different one from
"he was wrong".

The full reasoning is on the report page.

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



