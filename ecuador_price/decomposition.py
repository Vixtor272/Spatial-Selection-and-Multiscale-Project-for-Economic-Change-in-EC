#!/usr/bin/env python3
"""
decomposition.py
================
The one driver for the Ecuador multiscale Price decomposition, at ANY time
grain (monthly, quarterly, half-yearly, yearly).

The mathematics is not implemented here. It is the engine of Kemp, Fürsich,
Tannira & Bettencourt, "Spatial Selection and the Multiscale Dynamics of Urban
Change" (arXiv:2511.06165), file UrbanTraitSelection/calculation_scripts/
runPrice.py, imported as-is; nothing in it is modified. This file only turns
the Ecuadorian firm panel into the tables that engine expects, calls it once
per time window, and saves what it returns.

    Chicago (the paper)        Ecuador (this project)
    -------------------------  ---------------------------------------------
    household                  active formal firm (private corporation)
    household income           real total sales of the parish (2025 USD)
    block group < tract        parish < canton < province (+ country rollup)
      < community < county

Three decompositions ("paths") per window, same engine, same hierarchy
(parish < canton < province + country rollup):

    path  machinery              fitness / growth        trait (level)
    ----  ---------------------  ----------------------  --------------------
    pop   Price (rebalanced)     firm-count growth n'/n  log(real total sales)
    inc   Olley-Pakes style      real-sales log growth   real-sales LEVEL
    den   Olley-Pakes style      firm-count log growth   FIRM COUNT itself  n

The den path is the exact Olley-Pakes parallel of the inc path with the firm
count in place of sales: selection = cov(log firm growth, firm count) under
initial firm-share weights, DIVIDED by the group's mean firm count (the same
/mu normalization the inc path applies to sales). That mean firm count is the
size-biased mean sum_j (n_j/N) n_j = N x HHI -- the size of the parish in
which the AVERAGE FIRM sits ("experienced density") -- so den selection is
the per-year RELATIVE drift of that experienced density caused by
differential growth. It is dimensionless (per year), directly comparable to
the inc numbers. Selection < 0 = small parishes out-grow large ones
(dispersion of the firm mass); exactly zero under proportional (Gibrat)
growth, so the multiscale den terms are a Gibrat test by scale.

How the den path is computed without touching the engine
---------------------------------------------------------
The engine's two machineries are generic in the trait columns. We run the
engine a second time per window with the trait columns holding the firm
count (LogAvgInc* = log(firm count), so exp() reproduces the count exactly
the way exp(log sales) reproduces sales for the inc path). That second run's
_inc-named columns ARE the den path: they are renamed inc -> den and merged
into the first run's per-level tables, so every window exports ONE set of
CSVs (and one pickle) carrying the _pop, _inc and _den columns side by side
-- the _pop Price decomposition stays the headline result and _den simply
rides along like _inc does. The identity selection + transmission = total is
exact at every scale for all three paths (the engine's own SanChk columns).
The second run's _pop columns (a fitness-rebalanced density variant in raw
firm units) are dropped.

Time windows (the fields of Settings)
-------------------------------------
freq sets the period grain: monthly | quarterly | halfyearly | yearly.
step_mode sets how the comparison windows are formed:
    "rolling"        period t -> t+window_horizon, advancing the start by
                     window_step periods -> a time series of (overlapping)
                     windows. With window_horizon = one year of periods
                     (monthly=12, quarterly=4) this is a rolling YEAR-OVER-YEAR
                     series: seasonality cancels (same calendar period at both
                     ends) and the elapsed time is exactly 1 yr. THIS IS THE
                     RECOMMENDED MODE for sub-annual data (see WARNING below).
    "consecutive"    period t -> t+1 (Jan->Feb->...; 2017->2018->...),
                     plus the full first->last span.
    "year_over_year" same sub-period, next year (Q1 2017 -> Q1 2018 ...),
                     grouped by sub-period -- the quarterly seasonality design.
custom_windows adds arbitrary spans by period label (e.g. ("2017", "2019")).
Every term is divided by the window's elapsed time IN YEARS, so all numbers
are per-year rates, comparable across frequencies.

WARNING on sub-annual "consecutive" steps: a one-period step (e.g. one month)
is (i) annualised by dividing by ~1/12, which multiplies the raw change by ~12,
and (ii) dominated by the calendar season (the Dec->Jan sales cycle), not by
real growth -- selection over a single month is mostly seasonal noise, and one
month is too short to show firms/revenue actually moving. Use "rolling" with
window_horizon = 12 for a meaningful monthly time analysis.

Validity floor is an ANNUAL decision: the $1,000/yr floor defines the
(parish, year) universe on the annual table (it removes ~55 near-empty
"parishes" with absurd totals of about $5 a year; see
results/diagnostics/floor_sensitivity.csv -- the result does not depend on it).
A period cell inside a valid parish-year only needs positive, loggable values,
because a real parish can have a near-zero month or quarter for seasonal
reasons. For freq = "yearly" the two conditions coincide. Only survivors enter
each window (parishes valid at both endpoints).

Outputs -> results/decomposition/<freq>_<mode>_<y0>_<y1>/
    <window>/parish_<level>_exported_terms.csv   _pop, _inc AND _den columns
    <window>/all_decomposition_results.pkl       merged, all three paths
    decomposition_summary.csv        every window x path, tidy
    selection_by_step.csv            the step windows only, tidy

How to run
----------
The three runs the thesis is built on:
    python -m ecuador_price.decomposition --thesis-runs

One run with your own settings (see --help for every option):
    python -m ecuador_price.decomposition --freq yearly --mode consecutive
    python -m ecuador_price.decomposition --freq monthly --mode rolling \
        --year-start 2018 --year-end 2022

From Python (this is what the analysis scripts and notebooks do):
    from ecuador_price import decomposition as dec
    panel   = dec.load_real_panel()
    monthly = dec.monthly_table(panel)
    pp      = dec.parish_period_table(monthly, "yearly")
    valid   = dec.annual_valid_cells(monthly)
    results = dec.run_window(pp, valid, "2017_to_2023", (2017, 1), (2023, 1), "yearly")
"""

import argparse
from dataclasses import dataclass, field
import pickle
import re
import sys

import numpy as np
import pandas as pd

from ecuador_price import paths

# --------------------------------------------------------------------------- #
# Import the exact Chicago decomposition engine (the git submodule).
# --------------------------------------------------------------------------- #
if not (paths.ENGINE_DIR / "runPrice.py").exists():
    raise ImportError(
        "The engine is missing: UrbanTraitSelection/calculation_scripts/runPrice.py "
        "was not found. Fetch it with:  git submodule update --init")
sys.path.insert(0, str(paths.ENGINE_DIR))
import runPrice as rp                                    # noqa: E402

# --------------------------------------------------------------------------- #
# Settings -- everything you would change between runs.
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """What one run of the driver compares.

    freq             "monthly" | "quarterly" | "halfyearly" | "yearly"
    step_mode        "rolling" | "consecutive" | "year_over_year"
    year_start/end   first and last calendar year of the panel to use

    Rolling-window controls (used only when step_mode == "rolling"). A window
    compares period t to period t+window_horizon, advancing the start by
    window_step periods each time -> a proper time series of windows. Set
    window_horizon to ONE YEAR of periods (monthly=12, quarterly=4) so both
    ends are the same calendar period: seasonality cancels and elapsed time is
    exactly 1 yr, so the per-year rates are honest (no 1/12 blow-up).
    window_step=1 gives one point per period (rolling year-over-year);
    window_step=window_horizon gives non-overlapping windows.

    include_full_span  add the first->last window(s), tagged "headline"
    custom_windows     extra spans by period label, tagged "custom", e.g.
                       freq="yearly":  [("2017", "2019"), ("2020", "2023")]
                       freq="monthly": [("2018M01", "2019M01")]
    export_per_unit    write the engine's per-unit CSVs for every window (a
                       monthly rolling run writes 73 window folders; set False
                       if you only want the summary tables)
    """
    freq: str = "monthly"
    step_mode: str = "rolling"
    year_start: int = 2017
    year_end: int = 2023
    window_horizon: int = 12
    window_step: int = 1
    include_full_span: bool = True
    custom_windows: list = field(default_factory=list)
    export_per_unit: bool = True

    def run_name(self):
        """Folder name of this run, e.g. "monthly_rolling12_2017_2023"."""
        mode_tag = (f"rolling{self.window_horizon}" if self.step_mode == "rolling"
                    else self.step_mode)
        return f"{self.freq}_{mode_tag}_{self.year_start}_{self.year_end}"


# The three runs the thesis is built on.
THESIS_RUNS = [
    # rolling year-over-year, one point per month: the time series
    Settings(freq="monthly", step_mode="rolling", window_horizon=12, window_step=1),
    # one step per calendar year + the headline and the two sub-periods
    Settings(freq="yearly", step_mode="consecutive",
             custom_windows=[("2017", "2019"), ("2020", "2023")]),
    # each quarter against the same quarter a year later: the seasonality check
    Settings(freq="quarterly", step_mode="year_over_year"),
]

# Choices that are part of the definition of the analysis, not of one run.
FIRM_COUNT_METHOD = "mean"    # collapse monthly firm stocks within a period: mean|last|max
SALES_FLOOR       = 1000.0    # ANNUAL data-validity floor (real $/yr), see docstring

# The Ecuador hierarchy, base->top, and the DPA-prefix geography.
HIERARCHY  = ["parish", "canton", "province", "country"]
BASE_LEVEL = "parish"
CHILD_TO_PARENT = {"parish":   "ParentCanton",
                   "canton":   "ParentProvince",
                   "province": "ParentCountry"}
PARENT_PREFIX = {"canton": 4, "province": 2, "country": 0}

PERIODS_PER_YEAR = {"monthly": 12, "quarterly": 4, "halfyearly": 2, "yearly": 1}
FREQ_TAG         = {"monthly": "M",  "quarterly": "Q", "halfyearly": "H"}

# The three paths. After the den merge they all live in the same per-level
# tables, each under its own column suffix (_pop, _inc, _den).
PATH_SPECS = [
    ("pop", "firm_count_growth", "log_real_sales"),
    ("inc", "real_sales_growth", "real_sales_level"),
    ("den", "firm_count_growth", "firm_count_level"),
]


# --------------------------------------------------------------------------- #
# Period helpers: a period is a (year, sub) pair, labelled e.g. "2019M07".
# --------------------------------------------------------------------------- #
def period_label(year, sub, freq):
    """(2019, 7, "monthly") -> "2019M07";  (2019, 3, "quarterly") -> "2019Q3";
    (2019, 1, "yearly") -> "2019"."""
    if freq == "yearly":
        return str(year)
    width = 2 if freq == "monthly" else 1
    return f"{year}{FREQ_TAG[freq]}{sub:0{width}d}"


def parse_period(label, freq):
    """Inverse of period_label: "2019M07" -> (2019, 7)."""
    if freq == "yearly":
        return int(label), 1
    year, sub = label.split(FREQ_TAG[freq])
    return int(year), int(sub)


def period_time(year, sub, freq):
    """Decimal-year position of a period's start; differences give elapsed years."""
    return year + (sub - 1) / PERIODS_PER_YEAR[freq]


def all_periods(settings):
    """Ordered (year, sub) list covering year_start..year_end at settings.freq."""
    return [(y, s) for y in range(settings.year_start, settings.year_end + 1)
            for s in range(1, PERIODS_PER_YEAR[settings.freq] + 1)]


def generate_windows(settings):
    """[(name, p0, p1, window_type)] for one run: the step windows given by
    settings.step_mode, plus the full span ("headline") and any custom spans."""
    freq = settings.freq
    periods = all_periods(settings)
    windows = []
    if settings.step_mode == "rolling":
        h, s = settings.window_horizon, settings.window_step
        if h < 1 or s < 1:
            raise ValueError("window_horizon and window_step must be >= 1 period.")
        for i in range(0, len(periods) - h, s):
            windows.append((periods[i], periods[i + h], "step"))
        if settings.include_full_span and len(periods) > 1:
            # Longest span whose end shares the START's sub-period, so the
            # headline is season-matched too (monthly: 2017M01 -> 2023M01).
            p0 = periods[0]
            same_sub = [p for p in periods if p[1] == p0[1]]
            windows.append((p0, same_sub[-1], "headline"))
    elif settings.step_mode == "consecutive":
        for p0, p1 in zip(periods, periods[1:]):
            windows.append((p0, p1, "step"))
        if settings.include_full_span and len(periods) > 1:
            windows.append((periods[0], periods[-1], "headline"))
    elif settings.step_mode == "year_over_year":
        for sub in range(1, PERIODS_PER_YEAR[freq] + 1):
            for y in range(settings.year_start, settings.year_end):
                windows.append(((y, sub), (y + 1, sub), "step"))
            if settings.include_full_span and settings.year_end > settings.year_start:
                windows.append(((settings.year_start, sub),
                                (settings.year_end, sub), "headline"))
    else:
        raise ValueError('step_mode must be "rolling", "consecutive" '
                         'or "year_over_year".')
    for lab0, lab1 in settings.custom_windows:
        windows.append((parse_period(lab0, freq), parse_period(lab1, freq), "custom"))
    return [(f"{period_label(*p0, freq)}_to_{period_label(*p1, freq)}", p0, p1, wtype)
            for p0, p1, wtype in windows]


# --------------------------------------------------------------------------- #
# Step 1. Load the panel and deflate sales to constant-2025 USD.
# --------------------------------------------------------------------------- #
def load_real_panel():
    """Read the analysis table (one row per parish x CIIU section x month) and
    add ``real_sales`` = nominal sales x that year's IPC factor.

    The IPC factor turns that year's dollars into constant-2025 dollars (the
    2025 factor is exactly 1). It is the same for every parish in a year, so it
    shifts log(sales) by a per-year constant: selection (a within-year
    covariance) is unchanged and only the transmission and the total drift lose
    the inflation component.
    """
    df = pd.read_parquet(paths.FIRM_REVENUE_PARQUET,
                         columns=["anio", "mes", "cod_parr",
                                  "total_sales", "total_firms"])
    df["year"] = df["anio"].astype(int)

    ipc = pd.read_csv(paths.IPC_CSV)
    ipc.columns = ["year", "factor"]                 # header is Spanish ("año")
    factor = dict(zip(ipc["year"], ipc["factor"]))
    missing = sorted(set(df["year"].unique()) - set(factor))
    if missing:
        raise ValueError(f"No IPC factor for year(s) {missing}.")
    df["real_sales"] = df["total_sales"] * df["year"].map(factor)
    return df


# --------------------------------------------------------------------------- #
# Step 2. Collapse the monthly x CIIU grain to one row per (parish, period).
# --------------------------------------------------------------------------- #
def _collapse_firm_stock(monthly, keys):
    """Firms are a monthly stock: sum across CIIU sections within a parish-month
    (different firms), then collapse the months inside each key group by
    FIRM_COUNT_METHOD. Sales are a flow and just accumulate."""
    grp = monthly.groupby(keys)
    if FIRM_COUNT_METHOD == "mean":
        firms = grp["firms"].mean()
    elif FIRM_COUNT_METHOD == "last":
        firms = grp["firms"].last()
    elif FIRM_COUNT_METHOD == "max":
        firms = grp["firms"].max()
    else:
        raise ValueError('FIRM_COUNT_METHOD must be "mean", "last" or "max".')
    out = pd.concat([firms.rename("firms"), grp["sales"].sum()], axis=1)
    return out.reset_index()


def monthly_table(panel):
    """One row per (parish, year, month): firms summed across CIIU sections."""
    return (panel.assign(mes_i=panel["mes"].astype(int))
                 .groupby(["cod_parr", "year", "mes_i"], as_index=False)
                 .agg(firms=("total_firms", "sum"),
                      sales=("real_sales", "sum"))
                 .sort_values(["cod_parr", "year", "mes_i"]))


def parish_period_table(monthly, freq):
    """One row per (parish, period) at ``freq``: ``firms`` = the period's
    monthly firm stocks collapsed by FIRM_COUNT_METHOD, ``sales`` = the
    period's summed real sales, ``period`` = the label ("2019", "2019Q3", ...).

    Example: freq="quarterly" puts months 7, 8, 9 of 2019 in sub = 3, so that
    parish's row is the mean of three monthly firm counts and the sum of three
    months of sales, labelled "2019Q3".
    """
    ppy = PERIODS_PER_YEAR[freq]
    monthly = monthly.copy()
    monthly["sub"] = (monthly["mes_i"] - 1) * ppy // 12 + 1
    out = _collapse_firm_stock(monthly, ["cod_parr", "year", "sub"])
    out["period"] = [period_label(y, s, freq) for y, s in zip(out["year"], out["sub"])]
    return out


def annual_valid_cells(monthly, sales_floor=SALES_FLOOR):
    """(parish, year) cells with a positive firm stock and annual real sales
    above ``sales_floor`` -- the same universe whatever the time grain is."""
    annual = _collapse_firm_stock(monthly, ["cod_parr", "year"])
    ok = annual[(annual["firms"] > 0) & (annual["sales"] > sales_floor)]
    return set(zip(ok["cod_parr"], ok["year"].astype(int)))


def valid_parishes(pp, period, valid_cells):
    """Parishes usable in `period`: annually valid AND positive loggable values."""
    p = pp[pp["period"] == period]
    p = p[(p["firms"] > 0) & (p["sales"] > 0)]
    return {c for c, y in zip(p["cod_parr"], p["year"].astype(int))
            if (c, y) in valid_cells}


# --------------------------------------------------------------------------- #
# Step 3. Per-level census frames for one period, for a given trait.
#         trait = "log_sales" (pop + inc paths) or "firms" (den path).
# --------------------------------------------------------------------------- #
def _aggregate_up(child, prefix_width, level_name, trait):
    """Roll children up to the parent given by the DPA-code prefix (0 -> "EC").
    AvgLogInc (the covariance trait) is ALWAYS the firm-weighted mean of the
    children's AvgLogInc -- the recursive correction that makes the identity
    exact at every scale. LogAvgIncome (the level trait, _inc path only) is
    log of the parent's own total."""
    child = child.copy()
    child["parent"] = "EC" if prefix_width == 0 else child["code"].str[:prefix_width]
    child["w_trait"] = child["population"] * child["AvgLogInc"]
    agg = child.groupby("parent", sort=True).agg(
        population=("population", "sum"),
        sales=("sales", "sum"),
        w_trait=("w_trait", "sum"),
    )
    agg["AvgLogInc"]    = agg["w_trait"] / agg["population"]
    agg["LogAvgIncome"] = np.log(agg["sales"] if trait == "log_sales"
                                 else agg["population"])
    agg.index.name = level_name
    return agg.drop(columns="w_trait")


def build_levels_for_period(pp, period, keep_parishes, trait):
    """{level: DataFrame indexed by unit code} for one period, parish->country,
    restricted to the window's survivor set (same tightening as the headline:
    every parent trait is the survivor-weighted mean of exactly the units the
    decomposition runs on, so the nested selection telescopes exactly)."""
    p = pp[pp["period"] == period]
    p = p[p["cod_parr"].isin(keep_parishes)].copy()
    p["population"] = p["firms"].astype(float)
    if trait == "log_sales":
        p["AvgLogInc"]    = np.log(p["sales"])
        p["LogAvgIncome"] = np.log(p["sales"])
    elif trait == "firms":                       # den path: exp(LogAvgInc) = the
        p["AvgLogInc"]    = np.log(p["population"])  # firm count, exactly as
        p["LogAvgIncome"] = np.log(p["population"])  # exp(log sales) = sales for inc
    else:
        raise ValueError('trait must be "log_sales" or "firms".')
    parish = p.set_index("cod_parr")[["population", "sales",
                                      "AvgLogInc", "LogAvgIncome"]]

    levels = {"parish": parish}
    child = parish.reset_index().rename(columns={"cod_parr": "code"})
    for level_name in ["canton", "province", "country"]:
        agg = _aggregate_up(child, PARENT_PREFIX[level_name], level_name, trait)
        levels[level_name] = agg
        child = agg.reset_index().rename(columns={level_name: "code"})
    return levels


# --------------------------------------------------------------------------- #
# Step 4. Join initial+final into one wide panel per level (survivors only).
# --------------------------------------------------------------------------- #
def build_panel(incomes, level, key0, key1):
    df_i = incomes[key0][level]
    df_f = incomes[key1][level]
    common = df_i.index.intersection(df_f.index)   # survivors only, like Chicago
    if common.empty:
        return pd.DataFrame()

    codes = common.astype(str)
    out = pd.DataFrame(index=range(len(common)))
    out["UnitName"]          = codes.tolist()
    out["PopInitialN"]       = df_i.loc[common, "population"].values
    out["Population"]        = df_f.loc[common, "population"].values
    out["LogAvgIncInitialN"] = df_i.loc[common, "LogAvgIncome"].values
    out["LogAvgIncome"]      = df_f.loc[common, "LogAvgIncome"].values
    out["AvgLogIncInitialN"] = df_i.loc[common, "AvgLogInc"].values
    out["AvgLogInc"]         = df_f.loc[common, "AvgLogInc"].values
    out["PopInitialNR"]      = out["PopInitialN"]   # no separate "real" count
    out["PopulationR"]       = out["Population"]

    parent_col = CHILD_TO_PARENT.get(level)
    if parent_col:
        width = PARENT_PREFIX[{"ParentCanton": "canton",
                               "ParentProvince": "province",
                               "ParentCountry": "country"}[parent_col]]
        out[parent_col] = "EC" if width == 0 else [c[:width] for c in codes]

    out["Agg"] = level
    out["FirstYear"] = key0
    out["Year"] = key1
    return out


# --------------------------------------------------------------------------- #
# Step 5. Drive the engine: one run per trait set, per window.
# --------------------------------------------------------------------------- #
def run_engine(pp, survivors, key0, key1, elapsed_years, trait):
    """One full engine pass for one window and one trait set; returns the
    results dict {level: DataFrame}. elapsed_years may be fractional (a monthly
    step is 1/12) -- the engine only divides by it, so every term comes out as
    a per-YEAR rate."""
    incomes = {key0: build_levels_for_period(pp, key0, survivors, trait),
               key1: build_levels_for_period(pp, key1, survivors, trait)}

    # The engine reads a few settings as module globals; set them per run.
    # INITIAL/FINAL_YEAR only feed Chicago's split-tract step (never fires here).
    rp.CHILD_TO_PARENT_ID_COL_MAP = CHILD_TO_PARENT
    rp.INITIAL_YEAR, rp.FINAL_YEAR = key0, key1

    prepared = {}
    for level in HIERARCHY:
        raw = build_panel(incomes, level, key0, key1)
        prepared[level] = (rp.prepare_level_data(raw, level, elapsed_years,
                                                 CHILD_TO_PARENT)
                           if not raw.empty else pd.DataFrame())

    results = rp.calculate_multilevel_price_decomposition(
        prepared, BASE_LEVEL, HIERARCHY, CHILD_TO_PARENT, elapsed_years, incomes)
    return rp.calculate_ns_and_mss_metrics(
        results, HIERARCHY, CHILD_TO_PARENT, BASE_LEVEL)


# A lowercase "inc" delimited by underscores/string ends is the engine's
# income-machinery marker (AvgG_inc_canton, cum_sel_inc, ..._inc_ns). It never
# matches the capitalised income columns (LogAvgIncInitial_, ActualAvgInc*).
_INC_TOKEN = re.compile(r"(?<![A-Za-z0-9])inc(?![A-Za-z0-9])")


def merge_den_results(res_main, res_den):
    """Attach the den path to the main run's tables, level by level: every
    _inc-token column of the den (firm-count-trait) run -- the Olley-Pakes
    machinery, which for that run decomposes the firm count instead of sales
    -- is renamed inc -> den and merged on UnitName. The den run's _pop
    columns (a fitness-rebalanced density variant in raw firm units) and its
    descriptive columns (populations identical to the main run; the trait IS
    the firm count, already exported as Pop*) are dropped."""
    merged = {}
    for level, df_main in res_main.items():
        df_den = res_den.get(level)
        if df_main is None or df_main.empty or df_den is None or df_den.empty:
            merged[level] = df_main
            continue
        den_cols = [c for c in df_den.columns if _INC_TOKEN.search(c)]
        renamed = df_den[["UnitName"] + den_cols].rename(
            columns={c: _INC_TOKEN.sub("den", c) for c in den_cols})
        merged[level] = df_main.merge(renamed, on="UnitName", how="left")
    return merged


def export_results(results, out_dir):
    """Write parish_<level>_exported_terms.csv and all_decomposition_results.pkl
    exactly like the engine's _export_results, but keeping the _den columns too
    (the engine's column filter only knows _pop and _inc)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for level_name, df_lvl in results.items():
        if df_lvl is None or df_lvl.empty:
            continue
        potential = ["UnitName"] + list(CHILD_TO_PARENT.values())
        potential += [f"PopInitial_{level_name}", f"PopFinal_{level_name}",
                      f"LogAvgIncInitial_{level_name}", f"LogAvgIncFinal_{level_name}",
                      f"PopG_{level_name}", f"RelPopG_{level_name}",
                      f"AvgG_emp_{level_name}", f"RelAvgG_emp_{level_name}"]
        for path in ["pop", "den", "inc"]:
            potential += [f"AvgG_{path}_{level_name}", f"RelAvgG_{path}_{level_name}",
                          f"LHS_AvgG_{path}_{level_name}"]
        potential += [f"PopInitialR_{level_name}", f"PopFinalR_{level_name}"]
        for c in df_lvl.columns:
            if c.startswith(("Sel_", "Transmitted_")) and \
                    ("_pop" in c or "_inc" in c or "_den" in c):
                potential.append(c)
        for c in df_lvl.columns:
            if c.startswith("Cumulative_Sel_"):
                potential.append(c)

        existing = list(dict.fromkeys(c for c in potential if c in df_lvl.columns))
        raw_cols = [c for c in existing if not c.endswith("_mss") and "_ns" not in c]
        mss_cols = sorted(c for c in existing if c.endswith("_mss"))
        ns_cols = sorted(c for c in existing if "_ns" in c)
        final_cols = raw_cols + mss_cols + ns_cols
        if final_cols:
            df_lvl[final_cols].to_csv(
                out_dir / f"parish_{level_name}_exported_terms.csv", index=False)

    with open(out_dir / "all_decomposition_results.pkl", "wb") as f:
        pickle.dump({BASE_LEVEL: results}, f)


def run_window(pp, valid_cells, name, p0, p1, freq, out_dir=None):
    """Both engine passes for one window, merged into ONE results dict
    {level: DataFrame} whose tables carry the _pop, _inc and _den columns side
    by side.

    pp, valid_cells   from parish_period_table() and annual_valid_cells()
    name              label of the window, e.g. "2017_to_2023"
    p0, p1            its two ends as (year, sub) pairs, e.g. (2017, 1), (2023, 1)
    freq              the time grain ``pp`` was built at
    out_dir           if given, the per-unit CSVs and the pickle are written
                      to out_dir/<name>/; if None nothing is written
    """
    key0, key1 = period_label(*p0, freq), period_label(*p1, freq)
    elapsed = period_time(*p1, freq) - period_time(*p0, freq)
    print(f"\n{'='*12} window {name}  (elapsed {elapsed:.4f} yr) {'='*12}")

    valid0 = valid_parishes(pp, key0, valid_cells)
    valid1 = valid_parishes(pp, key1, valid_cells)
    survivors = valid0 & valid1
    print(f"  parishes: {len(valid0)} in {key0}, {len(valid1)} in {key1}, "
          f"{len(survivors)} survivors (used)")

    res_main = run_engine(pp, survivors, key0, key1, elapsed, "log_sales")
    res_den  = run_engine(pp, survivors, key0, key1, elapsed, "firms")
    results = merge_den_results(res_main, res_den)

    if out_dir is not None:
        export_results(results, out_dir / name)
    _verify_identity(results, name)
    return results


def _verify_identity(results, tag):
    """The multi-level identity via the engine's SanChk columns (pop and den
    paths), plus the national selection + transmission == total change."""
    problems = []
    for suffix in ["pop", "den"]:
        for parent in ["canton", "province", "country"]:
            df = results.get(parent)
            if df is None or df.empty:
                continue
            col = f"SanChk_MultiLevel_{suffix}_parish_to_{parent}"
            if col in df.columns:
                worst = df[col].abs().max()
                if worst > 1e-6:
                    problems.append(f"{col}={worst:.1e}")
        c = results.get("country", pd.DataFrame())
        if not c.empty:
            row = c.iloc[0]
            lhs = row.get(f"AvgG_{suffix}_country", np.nan)
            rhs = (row.get(f"Cumulative_Sel_{suffix}_country", np.nan)
                   + row.get(f"Transmitted_AvgG_parish_to_country_{suffix}", np.nan))
            if pd.notna(lhs) and pd.notna(rhs) and abs(lhs - rhs) > 1e-6:
                problems.append(f"country {suffix} sel+trans off by {lhs - rhs:.1e}")
    print(f"  identity check ({tag}): " + ("OK" if not problems else " ".join(problems)))


# --------------------------------------------------------------------------- #
# Step 6. Headline national numbers, every window x path, one tidy table.
# --------------------------------------------------------------------------- #
def national_summary_rows(results, name, p0, p1, window_type, freq):
    """One row per path (pop, inc, den) with the national terms of a window:
    total change, selection split by scale, transmission, and the residual of
    the identity total = selection + transmission."""
    key0, key1 = period_label(*p0, freq), period_label(*p1, freq)
    elapsed = period_time(*p1, freq) - period_time(*p0, freq)
    country = results.get("country", pd.DataFrame())
    if country.empty:
        return []
    row = country.iloc[0]

    rows = []
    for path_label, fitness, trait in PATH_SPECS:
        def get(col):
            return float(row[col]) if col in country.columns and pd.notna(row[col]) else np.nan

        total        = get(f"AvgG_{path_label}_country")
        sel_total    = get(f"Cumulative_Sel_{path_label}_country")
        sel_province = get(f"Sel_country_from_province_{path_label}")            # between provinces
        sel_canton   = get(f"Transmitted_Sel_province_to_country_{path_label}")  # cantons within province
        sel_parish   = get(f"Transmitted_Sel_canton_to_country_{path_label}")    # parishes within canton
        transmission = get(f"Transmitted_AvgG_parish_to_country_{path_label}")
        denom = sel_total if (pd.notna(sel_total) and sel_total != 0) else np.nan
        rows.append({
            "window": name, "window_type": window_type,
            "initial_period": key0, "final_period": key1,
            "initial_year": p0[0], "season": key0[4:] or "Y",   # sub-period tag
            "elapsed_years": elapsed,
            "path": path_label,
            "fitness": fitness,
            "trait": trait,
            "delta_trait_per_year":        total,
            "selection_total":             sel_total,
            "sel_between_provinces":        sel_province,
            "sel_canton_within_province":   sel_canton,
            "sel_parish_within_canton":     sel_parish,
            "transmission":                transmission,
            "share_between_provinces":      sel_province / denom,
            "share_canton_within_province": sel_canton / denom,
            "share_parish_within_canton":   sel_parish / denom,
            "selection_share_of_total":     (sel_total / total) if (pd.notna(total) and total != 0) else np.nan,
            "identity_residual":            total - (sel_total + transmission),
        })
    return rows


# --------------------------------------------------------------------------- #
# Step 7. One full run: every window of the settings, tables saved.
# --------------------------------------------------------------------------- #
def run(settings):
    """Run every window of ``settings`` and write the run folder. Returns the
    summary table (one row per window x path)."""
    freq = settings.freq
    out_root = paths.decomposition_run(settings.run_name())
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"=== Ecuador Price decomposition : {settings.run_name()} ===")
    print("paths: pop (log sales | firm growth), inc (sales level | sales growth), "
          "den (firm-count level | firm-count growth, Olley-Pakes parallel of inc)")

    panel = load_real_panel()
    monthly = monthly_table(panel)
    pp = parish_period_table(monthly, freq)
    valid_cells = annual_valid_cells(monthly)
    print(f"panel: {len(panel):,} rows -> {len(pp):,} parish-period cells; "
          f"{len(valid_cells):,} annually-valid (parish, year) cells "
          f"(>= ${SALES_FLOOR:,.0f}/yr).")

    summary = []
    for name, p0, p1, wtype in generate_windows(settings):
        results = run_window(pp, valid_cells, name, p0, p1, freq,
                             out_dir=out_root if settings.export_per_unit else None)
        summary.extend(national_summary_rows(results, name, p0, p1, wtype, freq))

    summary_df = pd.DataFrame(summary)
    summary_path = out_root / "decomposition_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Step-by-step rollup (the "step" windows only), one tidy table for all paths.
    step_cols = ["window", "path", "season", "initial_period", "final_period",
                 "delta_trait_per_year", "selection_total",
                 "sel_between_provinces", "sel_canton_within_province",
                 "sel_parish_within_canton", "transmission", "identity_residual"]
    by_step = (summary_df[summary_df["window_type"] == "step"][step_cols]
               .sort_values(["path", "initial_period"]))
    by_step_path = out_root / "selection_by_step.csv"
    by_step.to_csv(by_step_path, index=False, encoding="utf-8-sig")

    worst = summary_df["identity_residual"].abs().max()
    print(f"\nworst identity residual across all windows x paths: {worst:.1e}")

    show = ["window", "path", "delta_trait_per_year", "selection_total",
            "sel_between_provinces", "sel_canton_within_province",
            "sel_parish_within_canton", "transmission"]
    with pd.option_context("display.width", 220, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        head = summary_df[summary_df["window_type"] != "step"]
        if not head.empty:
            print("\n=== headline/custom windows, all paths (per year) ===")
            print(head[show].to_string(index=False))
        steps = by_step[by_step["path"] == "pop"]
        if len(steps) <= 30:
            print("\n=== step windows, pop path (per year) ===")
            print(steps[show].drop(columns="path").to_string(index=False))
        else:
            print(f"\n({len(steps)} step windows per path -> see {by_step_path.name})")

    print(f"\nsaved summary -> {summary_path}")
    print(f"saved by-step -> {by_step_path}")
    if settings.export_per_unit:
        print(f"per-unit CSVs -> {out_root}/<window>/parish_<level>_exported_terms.csv"
              f"  (_pop, _inc and _den columns together)")
    print("=== done ===")
    return summary_df


def main():
    defaults = Settings()
    parser = argparse.ArgumentParser(
        description="Multiscale Price decomposition of Ecuador's formal firms.")
    parser.add_argument("--thesis-runs", action="store_true",
                        help="run the three runs the thesis uses and ignore the other options")
    parser.add_argument("--freq", default=defaults.freq,
                        choices=["monthly", "quarterly", "halfyearly", "yearly"])
    parser.add_argument("--mode", dest="step_mode", default=defaults.step_mode,
                        choices=["rolling", "consecutive", "year_over_year"])
    parser.add_argument("--year-start", type=int, default=defaults.year_start)
    parser.add_argument("--year-end", type=int, default=defaults.year_end)
    parser.add_argument("--window-horizon", type=int, default=defaults.window_horizon,
                        help="rolling mode: periods spanned by each window")
    parser.add_argument("--window-step", type=int, default=defaults.window_step,
                        help="rolling mode: periods to advance the start each step")
    parser.add_argument("--no-full-span", action="store_true",
                        help="do not add the first->last headline window")
    parser.add_argument("--custom-window", nargs=2, action="append", default=[],
                        metavar=("START", "END"),
                        help='extra span by period label, e.g. --custom-window 2017 2019 '
                             '(may be repeated)')
    parser.add_argument("--summary-only", action="store_true",
                        help="skip the per-unit CSVs of every window")
    args = parser.parse_args()

    if args.thesis_runs:
        for settings in THESIS_RUNS:
            run(settings)
        return
    run(Settings(freq=args.freq, step_mode=args.step_mode,
                 year_start=args.year_start, year_end=args.year_end,
                 window_horizon=args.window_horizon, window_step=args.window_step,
                 include_full_span=not args.no_full_span,
                 custom_windows=[tuple(w) for w in args.custom_window],
                 export_per_unit=not args.summary_only))


if __name__ == "__main__":
    main()
