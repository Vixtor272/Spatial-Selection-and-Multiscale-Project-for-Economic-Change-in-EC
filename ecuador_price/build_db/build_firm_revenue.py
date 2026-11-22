#!/usr/bin/env python3
"""
build_firm_revenue.py
=====================
Build the analysis table: sales joined to the count of active firms, one row
per parish x CIIU section x month, keeping ONLY complete rows.

  * The SRI -> DPA parish crosswalk is written out here, step by step, instead of
    importing another module. You can read the whole build top to bottom.
  * After the merge, every row that still has a missing value -- an unmapped
    parish, a year with no firm counts, a sales/firm-count mismatch, or any other
    reason -- is REMOVED from the parquet and written to a CSV instead. Nothing is
    silently kept and nothing is silently lost.

Grain: one row per (anio, mes, cod_ciuu4, cod_parr, tipo).

Inputs (named in ecuador_price/paths.py)
  data/processed/df_ciuu_total_sales_purchases.csv   revenue, with SRI parish codes
                                                 (from prepare_sales_extract.py)
  data/raw/dpa.xlsx                              SRI -> DPA parish crosswalk source
  data/processed/active_firms_dpa.parquet        SRI active-firm counts, DPA parish
                                                 (from build_active_firms.py)

Outputs (data/processed/)
  firm_revenue_enriched_clean.parquet            complete rows only
  firm_revenue_enriched_dropped.csv              the removed rows + why they went

Run, after build_active_firms:
  python -m ecuador_price.build_db.build_firm_revenue
"""
import pandas as pd

from ecuador_price import paths

# --------------------------------------------------------------------------- #
# Parameters -- change these to re-point or re-window the build.
# --------------------------------------------------------------------------- #
REVENUE_CSV   = paths.SALES_CSV
DPA_XLSX      = paths.DPA_XLSX
FIRMS_PARQUET = paths.ACTIVE_FIRMS_PARQUET

OUT_PARQUET = paths.FIRM_REVENUE_PARQUET
DROPPED_CSV = paths.FIRM_REVENUE_DROPPED_CSV

TIPO     = "SDD_NO_PUB"                 # the only firm type both tables share
YEAR_MIN = 2015                         # keep rows whose anio is in
YEAR_MAX = 2025                         # [YEAR_MIN, YEAR_MAX] (inclusive)
KEYS     = ["anio", "mes", "cod_ciuu4", "cod_parr", "tipo"]   # the row grain

# Manual SRI -> DPA fixes for territorial reforms (Santa Elena 2007, Santo
# Domingo 2007, La Concordia 2013, cantonizations, SRI-only sub-parishes, and a
# couple of catalog typos). This is the same table the project keeps in
# ecuador_price/build_db/map_parroquias_sri_to_dpa.py, copied here so this script needs
# no other code file. SRI codes not listed here are resolved automatically below.
MANUAL_FIXES = {
    # --- Provincia de Santa Elena (creada 2007, antes Guayas) ---
    "1091750": "240150",  # SANTA ELENA (cabecera del nuevo canton 2401)
    "1091701": "240150",  # BALLENITA -> cabecera
    "1091702": "240150",  # SANTA ELENA (urbana) -> cabecera
    "1091751": "240151",  # ATAHUALPA
    "1091752": "240152",  # COLONCHE
    "1091753": "240153",  # CHANDUY
    "1091754": "240154",  # MANGLARALTO
    "1091755": "240155",  # SIMON BOLIVAR (JULIO MORENO)
    "1091756": "240156",  # SAN JOSE DE ANCON
    "1092650": "240250",  # LA LIBERTAD (canton 2402)
    "1091550": "240350",  # SALINAS (cabecera del nuevo canton 2403)
    "1091501": "240350",  # CARLOS ESPINOZA LARREA (urbana de Salinas) -> cabecera
    "1091502": "240350",  # GRAL ALBERTO GALLO (urbana de Salinas) -> cabecera
    "1091503": "240350",  # VICENTE ROCAFUERTE (urbana de Salinas) -> cabecera
    "1091515": "240351",  # ANCONCITO
    "1091552": "240352",  # JOSE LUIS TAMAYO (MUEY)
    # --- Provincia de Santo Domingo de los Tsachilas (creada 2007, antes Pichincha) ---
    "2170650": "230150",  # SANTO DOMINGO DE LOS COLORADOS (cabecera)
    "2170601": "230150",  # ABRAHAM CALAZACON (urbana) -> cabecera
    "2170602": "230150",  # BOMBOLI (urbana) -> cabecera
    "2170603": "230150",  # CHINGUILPE (urbana) -> cabecera
    "2170604": "230150",  # RIO TOACHI (urbana) -> cabecera
    "2170605": "230150",  # RIO VERDE (urbana) -> cabecera
    "2170606": "230150",  # SANTO DOMINGO DE LOS COLORADOS (urbana) -> cabecera
    "2170607": "230150",  # ZARACAY (urbana) -> cabecera
    "2170651": "230151",  # ALLURIQUIN
    "2170652": "230152",  # PUERTO LIMON
    "2170653": "230153",  # LUZ DE AMERICA
    "2170654": "230154",  # SAN JACINTO DEL BUA
    "2170655": "230155",  # VALLE HERMOSO
    "2170656": "230156",  # EL ESFUERZO
    "2170657": "230157",  # SANTA MARIA DEL TOACHI
    # --- La Concordia (Esmeraldas -> Santo Domingo, referendum 2013) ---
    "1080850": "230250",  # LA CONCORDIA
    "1080851": "230251",  # MONTERREY
    "1080852": "230252",  # LA VILLEGAS
    "1080853": "230253",  # PLAN PILOTO
    # --- Cantonizaciones / absorciones ---
    "1080151": "080650",  # ATACAMES: canton propio desde 1991
    "1090151": "090150",  # CHONGON: absorbida por el area urbana de Guayaquil
    "1120751": "121350",  # QUINSALOMA: canton propio desde 2007
    "3150755": "220453",  # SAN JOSE DE PAYAMINO: Napo -> Loreto, Orellana (1998)
    "2011551": "010852",  # EL CARMEN DE PIJILI -> Santa Isabel (DPA)
    # --- Subdivisiones solo-SRI sin parroquia DPA propia -> cabecera cantonal ---
    "1131251": "131250",  # SOSOTE -> Rocafuerte (Manabi)
    "2010455": "010450",  # LA PAZ -> Nabon (Azuay)
    # --- Correcciones de datos ---
    "20610":   "061050",  # CUMANDA: codigo truncado en el catalogo SRI (= 2061050)
    "3220254": "220254",  # TIPUTINI (Aguarico, Orellana): omitida en la hoja del crosswalk
}

# ========================================================================== #
# Step 1. Build the SRI -> DPA parish crosswalk from dpa.xlsx.
#
# The "Codigo Parroquias DPA" sheet holds two independent reference lists: a
# column of SRI parish codes and a column of the official DPA parish codes. They
# are NOT row-aligned, so we read the DPA column only as the set of codes that
# really exist, and we work out each SRI code's DPA equivalent from its digits:
#   - manual fix if the code was touched by a territorial reform (table above);
#   - else the last 6 digits, if that is itself a real DPA parish ("direct");
#   - else, for an urban sub-parish (last 2 digits < 50), the cantonal head,
#     i.e. the 4-digit canton prefix + "50".
# SRI codes that match none of these stay unmapped on purpose: their revenue
# rows will surface later as missing and be sent to the dropped CSV.
# ========================================================================== #
sheet = pd.read_excel(DPA_XLSX, sheet_name="Codigo Parroquias DPA")

# DPA codes come in as floats (e.g. 10150.0), so the leading zero is gone; go
# through int then pad back to 6 characters (10150 -> "010150").
valid_dpa = set(sheet["Codigo DPA"].dropna().astype("int64").astype(str).str.zfill(6))
sri_codes = sheet["Código SRI"].dropna().astype("int64").astype(str)

sri_to_dpa = {}
for code in sri_codes:
    if code in MANUAL_FIXES:
        sri_to_dpa[code] = MANUAL_FIXES[code]
    elif len(code) == 7 and code[1:] in valid_dpa:
        sri_to_dpa[code] = code[1:]                                  # direct
    elif len(code) == 7 and int(code[5:7]) < 50 and code[1:5] + "50" in valid_dpa:
        sri_to_dpa[code] = code[1:5] + "50"                         # urban -> cabecera

print(f"Crosswalk: {len(sri_to_dpa):,} SRI codes -> "
      f"{len(set(sri_to_dpa.values())):,} DPA parishes")

# ========================================================================== #
# Step 2. Read the revenue table and reduce it to the row grain.
# ========================================================================== #
cols = ["parroquia", "anio_fiscal", "mes_fiscal", "tipo", "ciiun4", "V_VTX", "V_X"]
rev = pd.read_csv(REVENUE_CSV, dtype=str, usecols=cols)
rev = rev[rev["tipo"] == TIPO].copy()

# money: text -> number; a blank sales/exports field counts as 0 in the sum.
rev["V_VTX"] = pd.to_numeric(rev["V_VTX"], errors="coerce").fillna(0.0)
rev["V_X"]   = pd.to_numeric(rev["V_X"],   errors="coerce").fillna(0.0)

# keys
rev["cod_ciuu4"] = rev["ciiun4"].str[0]                 # CIIU section letter (C110 -> C)
rev["cod_parr"]  = rev["parroquia"].str.strip().map(sri_to_dpa)   # SRI -> DPA (NaN if unmapped)
rev["anio"]      = rev["anio_fiscal"]
rev["mes"]       = rev["mes_fiscal"].str.zfill(2)       # '1' -> '01'

# keep only the requested year window
rev = rev[rev["anio"].astype(int).between(YEAR_MIN, YEAR_MAX)].copy()

# collapse to the grain, summing the two money columns (many SRI parishes share
# one DPA parish, so we must sum AFTER assigning cod_parr).
revenue = (rev.groupby(KEYS, as_index=False, dropna=False)
              .agg(total_sales=("V_VTX", "sum"), exports=("V_X", "sum")))

print(f"Revenue rows at the grain: {len(revenue):,}")

# ========================================================================== #
# Step 3. Read the firm counts and merge them onto the revenue rows.
# ========================================================================== #
firms = pd.read_parquet(FIRMS_PARQUET)[KEYS + ["total"]]
firms = firms.dropna(subset=["cod_parr"])               # ignore the few unmapped firm rows

merged = revenue.merge(firms, on=KEYS, how="left")
merged = merged.rename(columns={"total": "total_firms"})

# ========================================================================== #
# Step 4. Split complete rows from rows that still have a missing value.
# ========================================================================== #
data_cols = KEYS + ["total_sales", "exports", "total_firms"]
incomplete = merged[data_cols].isna().any(axis=1)

clean   = merged[~incomplete].copy()
dropped = merged[incomplete].copy()

clean["total_firms"] = clean["total_firms"].astype("int64")   # no NaN left, so plain int

# For each dropped row, record which variables were missing (for later analysis).
dropped["missing_variables"] = ""
for col in data_cols:
    dropped.loc[dropped[col].isna(), "missing_variables"] += col + ", "
dropped["missing_variables"] = dropped["missing_variables"].str.rstrip(", ")

# ========================================================================== #
# Step 5. Save and report.
# ========================================================================== #
clean.to_parquet(OUT_PARQUET, index=False)
dropped.to_csv(DROPPED_CSV, index=False, encoding="utf-8-sig")

print(f"\nRows in window:  {len(merged):,}")
print(f"  kept (clean):  {len(clean):,} ({len(clean)/len(merged):.2%})")
print(f"  dropped:       {len(dropped):,} ({len(dropped)/len(merged):.2%})")
print("\nWhy rows were dropped:")
print(dropped["missing_variables"].value_counts().to_string())
print(f"\nsaved clean   -> {OUT_PARQUET}")
print(f"saved dropped -> {DROPPED_CSV}")
print("\nClean sample:")
print(clean.head(8).to_string(index=False))
