"""
_ecuador_common.py
==================
Shared configuration and helpers for the Ecuador ports of the Chicago
UrbanTraitSelection/analysis_scripts plot suite.

Every script in this folder is a 1:1 port of the file with the same name in
UrbanTraitSelection/analysis_scripts (Kemp, Fürsich, Tannira & Bettencourt,
https://github.com/JordanK10/UrbanTraitSelection, commit 045a994), keeping the
original notation, plot structure and colors, but reading the Ecuador
Price-decomposition exports written by ecuador_price/decomposition.py:

    results/decomposition/<run>/<window>/parish_<level>_exported_terms.csv
    results/decomposition/<run>/<window>/all_decomposition_results.pkl

Level mapping (Chicago -> Ecuador)
----------------------------------
    bg  block group   -> parish     (base level of the decomposition)
    tr  tract         -> canton
    cm  community     -> province
    st  state (MSA)   -> country    (single national rollup row)
    ct  county        -> (grouping role only) province; Ecuador has one fewer
                         level than Chicago, so "grouped by county" becomes
                         "grouped by province" and "Cook County" becomes the
                         focal province (Pichincha by default).

"population" in the Chicago scripts = household count; here it is the number
of active formal firms.  "income" = E[y] household income; here it is real
total sales (constant-2025 USD).

Windows
-------
The decomposition was run for several windows.  Each plot script loops over
the requested windows (mirroring the Chicago scripts' loop over the
output_terms_null subdirectories) and writes into

    results/decomposition/<run>/plots/<window>/<plot_family>/

Usage of every script (from the project root):
    python -m ecuador_price.plots.<script>               # headline windows
    python -m ecuador_price.plots.<script> all           # every window
    python -m ecuador_price.plots.<script> 2019_to_2020  # specific window(s)
"""

import os

import numpy as np
import pandas as pd

from ecuador_price import paths

# --------------------------------------------------------------------------- #
# Paths -- the suite reads one run of the driver (ecuador_price/decomposition.py).
# Change RUN_NAME to plot a different run (e.g. a monthly one); the plots land
# inside the run folder itself, so runs never mix.
# --------------------------------------------------------------------------- #
RUN_NAME = "yearly_consecutive_2017_2023"
INPUT_ROOT = paths.decomposition_run(RUN_NAME)
OUTPUT_ROOT = INPUT_ROOT / "plots"

# Every window's exported CSVs carry THREE paths side by side, each under its
# own column suffix:
#   _pop  Price decomposition: fitness = firm-count growth, trait = log real
#         sales  (the headline result; log-points per year)
#   _inc  fitness = real-sales growth, trait = real-sales level  (per year)
#   _den  density: fitness = firm-count growth, trait = the firm count itself
#         (units = FIRMS per year, not log points)
HEADLINE_WINDOWS = ["2017_to_2023", "2017_to_2019", "2020_to_2023"]

# --------------------------------------------------------------------------- #
# Hierarchy (Chicago: ['bg', 'tr', 'cm', 'ct', 'st'])
# --------------------------------------------------------------------------- #
HIERARCHY_LEVELS = ["parish", "canton", "province", "country"]
BASE_ANALYSIS_LEVEL = "parish"

CHILD_TO_PARENT_ID_COL_MAP = {
    "parish": "ParentCanton",
    "canton": "ParentProvince",
    "province": "ParentCountry",
}

CODE_WIDTH = {"parish": 6, "canton": 4, "province": 2}

# Subscripts used in the LaTeX math labels (Chicago used cm/tr/bg).
MATH_SUBSCRIPT = {"parish": "par", "canton": "can", "province": "prov",
                  "country": "EC"}

# --------------------------------------------------------------------------- #
# Style — identical palette to the Chicago figures.
# --------------------------------------------------------------------------- #
CUSTOM_PURPLE = "#633673"
CUSTOM_ORANGE = "#E77429"
CUSTOM_GREY = "#3C3C3C"
OUTLINE_COLOR = "#3D3D3D"
PURPLE_DOT = "#caa8d6"
ORANGE_DOT = "#f2b58c"

RC_PARAMS = {"text.usetex": False, "font.family": "serif",
             "font.serif": ["Computer Modern Roman"]}

# --------------------------------------------------------------------------- #
# Geography names — DPA province codes.  The Cook County analog is the focal
# province: Pichincha (Quito).  Switch to "09" for Guayas (Guayaquil) if
# preferred.
# --------------------------------------------------------------------------- #
FOCAL_PROVINCE_CODE = "17"

PROVINCE_NAME = {
    "01": "Azuay", "02": "Bolívar", "03": "Cañar", "04": "Carchi",
    "05": "Cotopaxi", "06": "Chimborazo", "07": "El Oro", "08": "Esmeraldas",
    "09": "Guayas", "10": "Imbabura", "11": "Loja", "12": "Los Ríos",
    "13": "Manabí", "14": "Morona Santiago", "15": "Napo", "16": "Pastaza",
    "17": "Pichincha", "18": "Tungurahua", "19": "Zamora Chinchipe",
    "20": "Galápagos", "21": "Sucumbíos", "22": "Orellana",
    "23": "Santo Domingo de los Tsáchilas", "24": "Santa Elena",
}


def focal_province_name():
    return PROVINCE_NAME.get(FOCAL_PROVINCE_CODE, f"Province {FOCAL_PROVINCE_CODE}")


# --------------------------------------------------------------------------- #
# Window selection / iteration
# --------------------------------------------------------------------------- #
def available_windows():
    """Window subfolders of the master run that hold exported terms."""
    if not INPUT_ROOT.is_dir():
        return []
    return [d.name for d in sorted(INPUT_ROOT.iterdir())
            if d.is_dir() and (d / "parish_parish_exported_terms.csv").exists()]


def windows_from_argv(argv):
    """Default: the three headline windows.  'all' -> every window; otherwise
    any window names given on the command line."""
    args = [a for a in argv if not a.startswith("-")]
    avail = available_windows()
    if "all" in args:
        return avail
    named = [a for a in args if a in avail]
    if named:
        return named
    return [w for w in HEADLINE_WINDOWS if w in avail]


def iter_windows(argv):
    """Yield (window_name, input_dir, base_output_dir) as str paths, mirroring
    the Chicago scripts' (input_dir, base_output_dir) loop."""
    for window in windows_from_argv(argv):
        input_dir = str(INPUT_ROOT / window)
        base_output_dir = str(OUTPUT_ROOT / window)
        os.makedirs(base_output_dir, exist_ok=True)
        yield window, input_dir, base_output_dir


def window_years(window):
    """Parse a window name -> (initial_year, final_year). Handles the master
    labels ('2017_to_2023') and the legacy labels of the pre-master run
    ('2017_2023_all', '2019_2020')."""
    parts = [p for p in window.split("_") if p != "to"]
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None, None


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
_STR_COLS = ["UnitName", "ParentCanton", "ParentProvince", "ParentCountry"]


def terms_path(input_dir, level):
    return os.path.join(input_dir, f"parish_{level}_exported_terms.csv")


def load_terms(input_dir, level, add_parents=True):
    """Read one exported-terms CSV, keeping the DPA codes as zero-padded text.

    When ``add_parents`` is True every ancestor code the DPA prefix implies is
    attached (ParentCanton / ParentProvince / ParentCountry), so scripts can
    group any level by province the way the Chicago scripts group by county.
    """
    path = terms_path(input_dir, level)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={c: str for c in _STR_COLS})
    width = CODE_WIDTH.get(level)
    if width is not None and "UnitName" in df.columns:
        df["UnitName"] = df["UnitName"].str.strip().str.zfill(width)
    for col, w in [("ParentCanton", 4), ("ParentProvince", 2)]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.zfill(w)
    if add_parents and width is not None:
        if width > 4 and "ParentCanton" not in df.columns:
            df["ParentCanton"] = df["UnitName"].str[:4]
        if width > 2 and "ParentProvince" not in df.columns:
            df["ParentProvince"] = df["UnitName"].str[:2]
        if "ParentCountry" not in df.columns:
            df["ParentCountry"] = "EC"
    return df


def load_pickle(input_dir):
    """Load all_decomposition_results.pkl -> {'parish': {level: DataFrame}}."""
    import pickle
    path = os.path.join(input_dir, "all_decomposition_results.pkl")
    if not os.path.exists(path):
        print(f"Error: Pickle file not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def country_growth(input_dir):
    """National growth rates per path {'_pop': AvgG_pop_country, ...}.
    This is the Chicago MSA growth rate (AvgG_*_st) analog."""
    df = load_terms(input_dir, "country", add_parents=False)
    if df is None or df.empty:
        return {"_pop": np.nan, "_inc": np.nan, "_den": np.nan}
    row = df.iloc[0]
    return {"_pop": float(row.get("AvgG_pop_country", np.nan)),
            "_inc": float(row.get("AvgG_inc_country", np.nan)),
            "_den": float(row.get("AvgG_den_country", np.nan))}


def add_country_ns(df, input_dir):
    """Add missing country-normalized NS columns (the Chicago ``*_ns_st``).

    The Ecuador export only computed NS relative to the immediate parent, so
    e.g. ``Sel_canton_from_parish_pop_ns_country`` is absent at the canton
    level.  The engine's definition is simply (term / AvgG_<path>_country)*100,
    which we reproduce here from the national rollup row.
    """
    growth = country_growth(input_dir)
    df = df.copy()
    for col in list(df.columns):
        if not (col.startswith("Sel_") or col.startswith("Transmitted_")):
            continue
        if col.endswith("_mss") or "_ns" in col:
            continue
        path = next((p for p in ("_pop", "_inc", "_den") if col.endswith(p)), None)
        if path is None:
            continue
        ns_col = f"{col}_ns_country"
        if ns_col in df.columns:
            continue
        denom = growth[path]
        df[ns_col] = (df[col] / denom) * 100 if pd.notna(denom) and denom != 0 else np.nan
    return df


def province_label(code):
    """'01' -> 'Azuay'; unknown codes fall back to the code itself."""
    code = str(code).strip().zfill(2)
    return PROVINCE_NAME.get(code, f"Province {code}")
