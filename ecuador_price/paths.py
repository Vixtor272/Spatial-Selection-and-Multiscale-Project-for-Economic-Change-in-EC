"""
paths.py
========
The one place where every folder and input file of the project is named.
Scripts and notebooks import their locations from here, so nothing else in the
code contains a path, and moving a folder means editing one line.

    from ecuador_price import paths
    panel = pd.read_parquet(paths.FIRM_REVENUE_PARQUET)

Layout (relative to the project root):

    UrbanTraitSelection/   the decomposition engine of Kemp et al. (git submodule)
    data/raw/              files as received from SRI / INEC (not in the repo)
    data/processed/        tables built by ecuador_price/build_db (not in the repo)
    results/               everything the analysis writes: tables, figures, videos

Writing results somewhere else
------------------------------
Set the environment variable ECUADOR_RESULTS_DIR to send every output to another
folder. The tests and the "did the numbers move?" checks use it so that a trial
run never overwrites the real results:

    ECUADOR_RESULTS_DIR=/tmp/trial python -m ecuador_price.decomposition --freq yearly
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# The engine: UrbanTraitSelection (Kemp, Fürsich, Tannira & Bettencourt).
# Only calculation_scripts/runPrice.py and procHelpers.py are used.
# --------------------------------------------------------------------------- #
ENGINE_DIR = ROOT / "UrbanTraitSelection" / "calculation_scripts"

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# Raw files (see data/README.md for where each one comes from)
SRI_SALES_EXTRACT = (RAW / "SRI_2015-2025_respaldo"                     # the on-request sales extract
                         / "BDD_AGR_VENTAS_PASTUDILLO_TR1170120260413045.csv")
ACTIVE_FIRMS_RAW_DIR = RAW / "Estadisticas de Contribuyentes Activos"   # sri_activos_<year>.csv
IPC_CSV = RAW / "IPC_Ecuador.csv"                                       # yearly deflator, 2025 = 1
DPA_XLSX = RAW / "dpa.xlsx"                                             # official parish codes and names
CIIU_DICTIONARY_XLSX = RAW / "dic_ciiu.xlsx"
MUNICIPALITY_TABLES_XLSX = RAW / "Anexos y Tablas para Municipios.xlsx"
SERVICE_POINTS_XLSX = RAW / "manual_puntos_atencion_anexo.xlsx"
GEO_OVERRIDES_CSV = DATA / "geo_overrides.csv"                          # hand-edited, in the repo

# Shapefiles of the three administrative levels (INEC / CONALI)
SHAPEFILES = {
    level: RAW / f"ORGANIZACION_TERRITORIAL_DEL_ESTADO_{name}"
                 / f"ORGANIZACION TERRITORIAL DEL ESTADO {name}.shp"
    for level, name in [("province", "PROVINCIAL"),
                        ("canton", "CANTONAL"),
                        ("parish", "PARROQUIAL")]
}

# Processed tables (written by ecuador_price/build_db, in this order)
SALES_CSV = PROCESSED / "df_ciuu_total_sales_purchases.csv"      # company sales, SRI parish codes
SALES_DPA_CSV = PROCESSED / "df_ciuu_dpa.csv"                    # SALES_CSV with DPA parish codes
SRI_TO_DPA_CROSSWALK_CSV = PROCESSED / "sri_to_dpa_crosswalk.csv"
ACTIVE_FIRMS_PARQUET = PROCESSED / "active_firms_dpa.parquet"
CROSSWALK_AUDIT_CSV = PROCESSED / "parroquia_crosswalk_audit.csv"
FIRM_REVENUE_PARQUET = PROCESSED / "firm_revenue_enriched_clean.parquet"   # THE analysis table
FIRM_REVENUE_DROPPED_CSV = PROCESSED / "firm_revenue_enriched_dropped.csv"

# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
RESULTS = Path(os.environ.get("ECUADOR_RESULTS_DIR", ROOT / "results"))

DECOMPOSITION = RESULTS / "decomposition"            # one subfolder per run of the driver
DIAGNOSTICS = RESULTS / "diagnostics"
TRAIT_SPLIT = RESULTS / "trait_split"
ROBUSTNESS = RESULTS / "robustness_data_decisions"
FIRM_GROWTH_DISTRIBUTIONS = RESULTS / "firm_growth_distributions"
SELECTION_ANATOMY = RESULTS / "selection_anatomy"
RESILIENCE = RESULTS / "resilience"
STORY = RESULTS / "story"


def decomposition_run(run_name):
    """Folder of one driver run, e.g. decomposition_run("yearly_consecutive_2017_2023")."""
    return DECOMPOSITION / run_name
