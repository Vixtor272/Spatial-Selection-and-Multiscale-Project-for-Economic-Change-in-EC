# Plot suite (ported from Kemp et al.)

**Credit.** Every script here is a 1:1 port of the script with the same name in
`analysis_scripts/` of [UrbanTraitSelection](https://github.com/JordanK10/UrbanTraitSelection)
(commit `045a994`), the code of J. T. Kemp, L. Fürsich, E. Tannira and
L. M. A. Bettencourt, *Spatial Selection and the Multiscale Dynamics of Urban
Change*, [arXiv:2511.06165](https://arxiv.org/abs/2511.06165). The figure
designs, notation and colours are theirs; what changed is the input (Ecuadorian
firms instead of Chicago households), the level names and the added density
(`_den`) variants. Please cite their paper when using these figures.

The ports read the Ecuador firm-revenue Price decomposition produced by
`ecuador_price/decomposition.py`. Each script keeps the
source script's name, plot structure, notation (γ, NS, MSS, ρ, ω) and colors
(purple `#633673`, orange `#E77429`).

## Data flow

The suite reads one run of the driver, selected by
`_ecuador_common.RUN_NAME` (default `yearly_consecutive_2017_2023`):

```
inputs   results/decomposition/<run>/<window>/
             parish_{parish,canton,province,country}_exported_terms.csv
             all_decomposition_results.pkl
outputs  results/decomposition/<run>/plots/<window>/<plot_family>/
```

## The three paths

Every exported CSV carries three decompositions side by side, one column
suffix each: `_pop` (Price headline: fitness = firm-count growth, trait =
log real sales), `_inc` (Olley-Pakes style: real-sales growth × sales level,
selection normalized by the mean level) and `_den` (the exact Olley-Pakes
parallel of `_inc` with the **firm count** in place of sales: firm-count
growth × firm-count level, normalized by the mean firm count = N×HHI, the
"experienced density"). All three are per-year rates; den is dimensionless
and directly comparable to inc. Wherever a script draws an `_inc` variant of
a plot it also draws the analogous `_den` variant (labels say "Density",
files say `den`/`density`). Two scatter families currently have no den
variant (`diff_scatters.py`, `specialty_scatter.py`) — they mix paths on one
axis and were kept two-path by design.

## Level / concept mapping (Chicago → Ecuador)

| Chicago                      | Ecuador                                |
| ---------------------------- | -------------------------------------- |
| block group`bg` (base)     | parish (base)                          |
| tract`tr`                  | canton                                 |
| community`cm`              | province                               |
| state/MSA`st` (1 row)      | country (1 row)                        |
| county`ct` (grouping)      | province grouping (`ParentProvince`) |
| Cook County                  | focal province (Pichincha,`17`)      |
| households ("population")    | active formal firms                    |
| household income             | real total sales (constant-2025 USD)   |
| `*_ns_st` (MSA-normalized) | `*_ns_country` (country-normalized)  |

Ecuador has one level fewer than Chicago, so summary heatmaps have 4 columns
instead of 5. The export lacks `*_ns_country` at the canton level; scripts
derive it as `term / AvgG_<path>_country × 100` (the engine's own formula) via
`_ecuador_common.add_country_ns`.

## Running

```bash
# from the project root; default = 3 headline windows
python -m ecuador_price.plots.run_all

# every window, incl. the year-by-year ones
python -m ecuador_price.plots.run_all all
# specific windows
python -m ecuador_price.plots.run_all 2019_to_2020
```

Every script also runs standalone with the same window arguments.

## Script map

| Script (same name as Chicago)                | What it draws for Ecuador                                                                       | Output folder                                 |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `createHeatmaps.py`                        | Country + Pichincha combined MSS/NS summary heatmaps; MSS variation summaries                   | `heatmaps_local_dominance/country_summary/` |
| `selection_summary_heatmaps.py`            | 3×4 ⟨ρ⟩ / σ(ρ) / ω heatmaps, pop & income paths                                          | `heatmaps_local_dominance/country_summary/` |
| `msa_normalized_selection_summary.py`      | Country-normalized selection intensity mean/std heatmap                                         | `heatmaps_local_dominance/country_summary/` |
| `histogram_data.py`                        | Growth-rate & initial-income histograms with t / normal fits (parish, canton, province)         | `histogram_plots/`                          |
| `specialty_histogram2.py`                  | NS (skew-t fit) and MSS (gamma fit) histograms, canton vs province overlay, linear & log scale  | `specialty_histograms2/`                    |
| `simpleScatters.py`                        | 10 growth-rate scatter types × {canton, province}                                              | `simple_scatter_plots/`                     |
| `diff_scatters.py`                         | 6 growth-rate-difference scatter types × {canton, province}                                    | `diff_scatter_plots/`                       |
| `specialty_scatter.py`                     | γ^i / γ^p vs γ with connecting lines and weighted CIs; full country + per-province (cantons) | `specialty_scatter_plots/`                  |
| `inequality_scatter.py`                    | Relative growth vs cumulative NS_ec, purple-white-orange colorbar                               | `inequality_scatter_plots/`                 |
| `sanity_check_scatter.py`                  | Decomposition LHS vs RHS identity checks                                                        | `sanity_check_{canton,province}/`           |
| `selection_vs_population_plot.py`          | log Σ\|Sel\| vs log firm stock by province (canton + province levels)                          | `specialty_scatter_plots/`                  |
| `selection_vs_population_plot_weighted.py` | per-firm weighted variant                                                                       | `specialty_scatter_plots/`                  |
| `scaling_analysis_summary.py`              | 3-panel scaling of canton selection aggregated to provinces                                     | `specialty_scatter_plots/`                  |
| `selection_concentration_curves.py`        | Concentration curves: per-province, unified pos/neg/abs, canton-vs-province                     | `concentration_curves/`                     |
| `county_contribution_pie_charts.py`        | Province contribution / density / firm-share pies                                               | `county_contribution_pies/`                 |
| `variance_by_level.py`                     | Mean ±1 SD of growth rates by aggregation level                                                | `variance_plots/`                           |
| `gini_inequality_scatter.py`               | Cumulative/transmitted/selection NS_ec vs within-province Gini change                           | `inequality_scatter_plots/`                 |
| `gini_scatter.py`                          | WLS-fit, median-income-bifurcated Gini scatters + summary heatmap                               | `gini_scatter_plots/`                       |
| `population_sorting_analysis.py`           | Canton selection NS vs within-canton Gini change (Census-API rent/Theil parts not portable)     | `inequality_scatter_plots/`                 |
| `population_turnover_analysis.py`          | Firm-stock turnover tables per level and per province (CSV + console)                           | `population_turnover/`                      |

Notes on deliberate adaptations (beyond the level renames):

- **Gini incomes**: Chicago's `exp(LogAvgInc)` is a per-household mean; the
  Ecuador trait is log *total* sales, so Gini uses per-firm mean sales
  `exp(LogAvgInc)/firms`, weighted by firm counts.
- **Hardcoded axis limits** in the Chicago scripts (tuned to ±0.09/yr income
  growth) are data-driven here — Ecuador sales growth spans roughly −0.4 to
  +0.7/yr.
- `specialty_scatter.py` filters units by `MIN_INITIAL_FIRMS = 10` (Chicago:
  5,000 households).
- The focal province is set in `_ecuador_common.FOCAL_PROVINCE_CODE`
  (`"17"` Pichincha; use `"09"` for Guayas).
- `specialty_critique.txt` in the Chicago folder documents a fitting bug that
  is already fixed in `specialty_histogram2.py` (MLE on raw data); nothing to
  port.
