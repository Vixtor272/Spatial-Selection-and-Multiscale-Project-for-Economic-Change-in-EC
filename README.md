# Selection and Transmission in Space

**A multiscale Price-equation decomposition of Ecuador's formal economy, 2017–2023.**
Undergraduate thesis of Victor Bastidas, Universidad San Francisco de Quito (USFQ), 2026.
Thesis tutor: Pablo Astudillo Estévez. Mentor: Jordan T. Kemp.

When Ecuador's formal economy grows, does it grow because places improve on their
own (*transmission*) or because firms are reallocated between places (*selection*)?
This repository splits the change of the country's average exactly into those two
parts, and splits selection again by scale: between provinces, between cantons
within a province, and between parishes within a canton.

Headline result, population path, 2017 → 2023, per year
(`results/decomposition/yearly_consecutive_2017_2023/decomposition_summary.csv`):

| total change | selection | transmission | selection between provinces |
| ---: | ---: | ---: | ---: |
| +0.0226 | −0.0270 | +0.0496 | −0.0227 (84 % of selection) |

The two theses explain what these numbers mean and how far they can be trusted:
[`thesis/es/tesis_bastidas.pdf`](thesis/es/tesis_bastidas.pdf) (Spanish, USFQ format, the
full version) and [`thesis/en/thesis_bastidas.pdf`](thesis/en/thesis_bastidas.pdf)
(English, IEEE format).

## Credit: the engine is not mine

The decomposition itself is computed by the code of

> J. T. Kemp, L. Fürsich, E. Tannira and L. M. A. Bettencourt,
> *Spatial Selection and the Multiscale Dynamics of Urban Change*,
> [arXiv:2511.06165](https://arxiv.org/abs/2511.06165).
> Code: <https://github.com/JordanK10/UrbanTraitSelection>

which they wrote for household income in Chicago neighbourhoods. This project uses
that code **unmodified**, at commit `045a994`, as a git submodule in
[`UrbanTraitSelection/`](UrbanTraitSelection). Their repository is linked, not
copied: it carries no licence, so its files are not redistributed here.

- `ecuador_price/decomposition.py` imports two of their files,
  `calculation_scripts/runPrice.py` and `procHelpers.py`, and calls them once per
  time window. All the algebra is theirs.
- `ecuador_price/plots/` is a port of their `analysis_scripts/` figure suite to the
  Ecuadorian data. Figure designs, notation and colours are theirs; each file names
  the script it was ported from.

What this project adds sits outside the engine: the Ecuadorian panel (firms, sales,
parish < canton < province), the density path (a second run of the engine with the
firm count as the trait), the choice of time windows, and the analysis of the
results. **If you use this repository, please cite their paper** as well as the
thesis (see [`CITATION.cff`](CITATION.cff)).

## What is in the repository

```
UrbanTraitSelection/   the engine of Kemp et al. (git submodule, commit 045a994)
ecuador_price/         the code of this thesis
    paths.py               every folder and input file, named once
    style.py               every colour, the map scale and the fonts, named once
    build_db/              raw SRI files -> the analysis table
    decomposition.py       the driver: prepares the panel and calls the engine
    analysis/              small tables and summary figures behind the text (trait split, robustness, outliers)
    plots/                 figure suite ported from Kemp et al.
    videos/                the Loja zoom animations
notebooks/             the analyses and figures of each thesis section
tests/                 checks that run without any data
data/                  how to obtain the data (the data itself is not included)
results/               tables and figures produced by the code above
thesis/                es/ (Spanish), en/ (English), figures/ (shared by both)
```

Three ways of looking at each window ("paths"), all from the same engine:

| path | who grows (fitness) | what is averaged (trait) |
| --- | --- | --- |
| `pop` | number of firms | log of real total sales |
| `inc` | real sales | level of real sales |
| `den` | number of firms | the number of firms itself |

## Setup

```bash
git clone --recursive <URL of this repository>
cd <folder>
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

`--recursive` also downloads the engine. If you cloned without it, run
`git submodule update --init`. The results were produced with Python 3.13; the exact
package versions are in [`pyproject.toml`](pyproject.toml).

Run every command below **from the project root**. That is all the code needs to
find itself, so it also works if the `pip install -e` step is skipped and only the
dependencies are installed.

## Check that everything works (no data needed)

```bash
.venv/bin/python -m pytest
```

The tests confirm that the engine is the exact pinned version, and that on an
invented economy the identities the thesis relies on hold to machine precision:
total = selection + transmission, selection = the sum of its three scales, zero
selection under proportional growth, and the exact split of the trait into firm
count and sales per firm. One more test recomputes the headline numbers and is
skipped when the data is not on the machine.

## Reproduce the results

The data is not in the repository; [`data/README.md`](data/README.md) explains what
each file is and how to obtain it. With the files in `data/raw/`:

```bash
# 1. build the analysis table (parish x CIIU section x month: sales and firms)
python -m ecuador_price.build_db.prepare_sales_extract
python -m ecuador_price.build_db.build_active_firms
python -m ecuador_price.build_db.build_firm_revenue
python -m ecuador_price.build_db.map_parroquias_sri_to_dpa   # CIIU 4-digit table, used by the resilience notebooks

# 2. the decomposition: the three runs the thesis uses
python -m ecuador_price.decomposition --thesis-runs

# 3. tables behind the text
python -m ecuador_price.analysis.trait_split_table
python -m ecuador_price.analysis.robustness_data_decisions
python -m ecuador_price.analysis.diagnose_price_outliers
python -m ecuador_price.analysis.other_taxpayer_types      # why private companies only (reads the raw files)

# 4. figures
python -m ecuador_price.analysis.decomposition_figures      # where the change comes from: 2017-2023 and year by year
python -m ecuador_price.plots.run_all all
#    then run the notebooks in notebooks/, resilience_analysis.ipynb first
#    (two others read the table it writes)

# 5. copy the figures the theses use and recompile them (needs a LaTeX engine, e.g. tectonic)
python thesis/sync_figures.py
(cd thesis/en && tectonic -X compile thesis_bastidas.tex)
(cd thesis/es && tectonic tesis_bastidas.tex)
```

Any other window is one command, for example
`python -m ecuador_price.decomposition --freq yearly --mode consecutive --year-start 2019 --year-end 2022`
(`--help` lists every option).

| notebook | thesis section |
| --- | --- |
| `story_ecuador_formal_economy` | the two margins (firms and sales), scale scatters |
| `decomp_monthly_analysis_pop` / `_inc` / `_den` | the monthly time series of each path |
| `decomp_season_data_analysis` | seasonality, growth distributions, anatomy of the 2019–2020 inversion |
| `masked_growth_multiscale_maps` | when the average is not a description (masked growth, Loja) |
| `resilience_analysis`, `metro_regular_economy_resilience` | Quito, Guayaquil and Cuenca through the pandemic |
| `mathematical_proof_preq` | selection and transmission are not additive over time; their sum is |

## Results

`results/` holds what the code writes: the per-window tables of every run
(`decomposition/<run>/<window>/parish_<level>_exported_terms.csv`), the summaries,
and the figures. Left out of the repository because of their size, and regenerated
by the code: the engine's `.pkl` files, the Loja videos (`.mp4`) and the full-page
vector maps of the masked-growth cases (their `.png` versions are included).

## Data and rights

The sales extract was delivered by Ecuador's tax authority (SRI) on request and is
not redistributed. No licence has been chosen for this repository yet: until one is
added, all rights are reserved by the author, and the engine in
`UrbanTraitSelection/` belongs to its authors.
