#!/usr/bin/env python3
"""
build_active_firms.py
=====================
Normalize the yearly SRI "Estadisticas de Contribuyentes Activos" files into a
single tidy parquet, one row per (year, month, CIIU section, DPA parish, firm
type) with the firm count.

Output columns
--------------
    anio      str   fiscal year, e.g. "2017" .. "2025"
    mes       str   month number, "01" .. "12"
    cod_ciuu4 str   CIIU rev.4 section letter (A, B, ...), from CODIGO_OPERA_FAMILIA
    cod_parr  str   6-digit INEC-DPA parish code, zero padded (e.g. "010150")
    tipo      str   firm type; only SECTOR PRIVADO is kept, relabeled "SDD_NO_PUB"
    total     int   number of active firms in that cell

Why each design choice
----------------------
* Encoding: the raw files are latin-1 (single-byte tildes/Ñ such as 0xD1=Ñ).
  We read latin-1 and then strip accents only for *matching*, so codification of
  tildes/Ñ never breaks a join.
* Delimiter: 2017-2020 use ';', 2021-2025 use '|'. We sniff the header line.
* Parish code (cod_parr): the raw files have NO parish code, only the name trio
  (provincia, canton, parroquia). The SAME parish name can appear in different
  cantons, so all three names are needed. We resolve the trio to the official
  INEC-DPA parish code through two reference files the project already curates:
      1. "Mapeo Parroquias"  (Anexos y Tablas para Municipios.xlsx)
         trio of names  ->  6-digit SRI parish code
      2. "BASE"              (manual_puntos_atencion_anexo.xlsx)
         6-digit SRI parish code  ->  DPA parish code (DPA_PARROQ)
  Step 2 already bakes in (a) the SRI->DPA territorial reforms and (b) the
  collapse of SRI urban sub-parishes into their cantonal head ("urbana a
  cabecera"). Following it keeps cod_parr identical in meaning to the project's
  existing `codigo_dpa`.
* Many SRI sub-parishes collapse onto one DPA parish, so after assigning
  cod_parr we GROUP BY the output keys and SUM `total` (never sum before the
  collapse, or a parish would be counted several times).
* Resolution precedence per trio (see resolve_parish):
      override  > mapeo > name > cabecera > unresolved
  `override` and the canton aliases live in an editable CSV (geo_overrides.csv);
  `name`/`cabecera` are the documented automatic fallbacks. Every trio and the
  method used is written to an audit CSV so the whole crosswalk can be reviewed.

Run
---
    python -m ecuador_price.build_db.build_active_firms
"""

import glob
import os
import re
import unicodedata

import pandas as pd

from ecuador_price import paths

# --------------------------------------------------------------------------- #
# Paths (all named in ecuador_price/paths.py)
# --------------------------------------------------------------------------- #
RAW_DIR = str(paths.ACTIVE_FIRMS_RAW_DIR)
MAPEO_XLSX = paths.MUNICIPALITY_TABLES_XLSX
PUNTOS_XLSX = paths.SERVICE_POINTS_XLSX
OVERRIDES_CSV = paths.GEO_OVERRIDES_CSV
OUT_PARQUET = paths.ACTIVE_FIRMS_PARQUET
AUDIT_CSV = paths.CROSSWALK_AUDIT_CSV

SUBTIPO_KEEP = "SECTOR PRIVADO"   # the only DESCRIPCION_SUBTIPO we keep
TIPO_LABEL = "SDD_NO_PUB"         # how SECTOR PRIVADO is relabeled in the output

# Columns we actually need from each raw yearly file.
RAW_COLS = [
    "DFC_ANIO_PK", "DFC_DESCRIPCION_MES", "CODIGO_OPERA_FAMILIA",
    "DESCRIPCION_SUBTIPO", "DESCRIPCION_PROVINCIA", "DESCRIPCION_CANTON",
    "DESCRIPCION_PARROQUIA", "TOTAL",
]


# --------------------------------------------------------------------------- #
# Text matching helper
# --------------------------------------------------------------------------- #
def norm(text) -> str:
    """Uppercase, drop accents/tildes (Ñ->N), collapse spaces, drop a leading '*'.

    Used only to compare names across files; the '*' prefix marks superseded
    rows in the manual_puntos sheet.
    """
    if text is None or (isinstance(text, float)):
        return ""
    s = str(text).strip().upper().lstrip("*").strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------- #
# Reference tables for the parish crosswalk
# --------------------------------------------------------------------------- #
def load_sri_to_dpa() -> dict:
    """6-digit SRI parish code -> DPA parish code, from manual_puntos 'BASE'.

    Rows whose DPA_PARROQ is blank are superseded SRI codes (marked '*'); we skip
    them so only canonical assignments survive.
    """
    base = pd.read_excel(PUNTOS_XLSX, sheet_name="BASE", dtype=str)
    base.columns = [c.strip() for c in base.columns]
    sri6 = (base["CODIGO PROVINCIA"].str.zfill(2)
            + base["CODIGO CANTON"].str.zfill(2)
            + base["CODIGO PARROQUIA"].str.zfill(2))
    base = base.assign(sri6=sri6, desc=base["DESCRIPCIÓN DE LA PARROQUIA"])
    good = base[base["DPA_PARROQ"].notna()]
    sri2dpa = dict(zip(good["sri6"], good["DPA_PARROQ"]))
    # (canton prefix, normalized SRI parish name) -> DPA code, for the name fallback
    name2dpa = {(s[:4], norm(d)): code
                for s, d, code in zip(good["sri6"], good["desc"], good["DPA_PARROQ"])}
    return sri2dpa, name2dpa


def load_trio_to_dpa(sri2dpa: dict):
    """Build, from 'Mapeo Parroquias':
        trio2dpa      normalized "prov|canton|parr" -> DPA code
        canton_prefix (norm prov, norm canton)      -> 4-digit DPA prefix
    A trio with several SRI codes (a cantonal seat coded twice) is resolved to the
    first of its codes that has a DPA equivalent; by construction they agree.
    """
    m = pd.read_excel(MAPEO_XLSX, sheet_name="Mapeo Parroquias", dtype=str)
    m.columns = [c.strip() for c in m.columns]
    p = m["DESCRIPCION PROVINCIA"].map(norm)
    c = m["DESCRIPCION CANTON"].map(norm)
    a = m["DESCRIPCION PARROQUIA"].map(norm)
    m = m.assign(np=p, nc=c, key=p + "|" + c + "|" + a)

    trio2dpa = {}
    for key, codes in m.groupby("key")["CODIGO PARROQUIA"]:
        hit = [sri2dpa[code] for code in codes if code in sri2dpa]
        if hit:
            trio2dpa[key] = hit[0]

    canton_prefix = {}
    for (np_, nc_), codes in m.groupby(["np", "nc"])["CODIGO PARROQUIA"]:
        canton_prefix[(np_, nc_)] = codes.str[:4].mode().iloc[0]
    return trio2dpa, canton_prefix


def load_overrides():
    """Read geo_overrides.csv -> (canton_aliases, trio_codes), both keyed normalized.

        canton_aliases  (norm prov_in, norm canton_in) -> (norm prov_out, norm canton_out)
        trio_codes      "prov|canton|parr" (normalized) -> DPA code
    """
    ov = pd.read_csv(OVERRIDES_CSV, dtype=str).fillna("")
    canton_aliases, trio_codes = {}, {}
    for r in ov.itertuples(index=False):
        if r.tipo == "canton_alias":
            canton_aliases[(norm(r.prov_in), norm(r.canton_in))] = (norm(r.prov_out), norm(r.canton_out))
        elif r.tipo == "trio_code":
            key = norm(r.prov_in) + "|" + norm(r.canton_in) + "|" + norm(r.parroquia_in)
            trio_codes[key] = str(r.cod_parr).strip()
    return canton_aliases, trio_codes


def make_resolver(sri2dpa, name2dpa, trio2dpa, canton_prefix, canton_aliases, trio_codes):
    """Return resolve(prov, canton, parr) -> (cod_parr or None, method)."""
    valid_dpa = set(sri2dpa.values())

    def resolve(prov, canton, parr):
        p, c, a = norm(prov), norm(canton), norm(parr)
        # 1. explicit trio override (territorial reforms, etc.)
        direct = trio_codes.get(f"{p}|{c}|{a}")
        if direct:
            return direct, "override"
        # 2. rename the canton to its Mapeo/DPA spelling if needed
        p, c = canton_aliases.get((p, c), (p, c))
        # 3. direct trio match in Mapeo
        code = trio2dpa.get(f"{p}|{c}|{a}")
        if code:
            return code, "mapeo"
        # 4. same parish name inside the canton (manual_puntos)
        prefix = canton_prefix.get((p, c))
        if prefix and (prefix, a) in name2dpa:
            return name2dpa[(prefix, a)], "name"
        # 5. SRI urban sub-parish with no own DPA code -> cantonal head (XX..50)
        if prefix and prefix + "50" in valid_dpa:
            return prefix + "50", "cabecera"
        return None, "unresolved"

    return resolve


# --------------------------------------------------------------------------- #
# Raw file reading
# --------------------------------------------------------------------------- #
def sniff_sep(path: str) -> str:
    """';' for 2017-2020, '|' for 2021-2025 — decided from the header line."""
    with open(path, "rb") as fh:
        head = fh.readline()
    return ";" if head.count(b";") >= head.count(b"|") else "|"


def read_one_year(path: str) -> pd.DataFrame:
    """Read one yearly file, keep SECTOR PRIVADO, return the tidy pre-crosswalk frame."""
    df = pd.read_csv(path, sep=sniff_sep(path), dtype=str, encoding="latin-1",
                     keep_default_na=False, usecols=RAW_COLS)
    df = df[df["DESCRIPCION_SUBTIPO"] == SUBTIPO_KEEP].copy()
    out = pd.DataFrame({
        "anio": df["DFC_ANIO_PK"].str.strip(),
        "mes": df["DFC_DESCRIPCION_MES"].str.extract(r"(\d{2})", expand=False),
        "cod_ciuu4": df["CODIGO_OPERA_FAMILIA"].str.strip(),
        "tipo": TIPO_LABEL,
        "total": df["TOTAL"].astype(int),
        "prov": df["DESCRIPCION_PROVINCIA"],
        "canton": df["DESCRIPCION_CANTON"],
        "parroquia": df["DESCRIPCION_PARROQUIA"],
    })
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "sri_activos_*.csv")))
    print(f"Reading {len(files)} yearly files ...")
    raw = pd.concat([read_one_year(f) for f in files], ignore_index=True)
    print(f"  {len(raw):,} SECTOR PRIVADO rows across {raw['anio'].nunique()} years")

    # Build the resolver from the reference files + overrides.
    sri2dpa, name2dpa = load_sri_to_dpa()
    trio2dpa, canton_prefix = load_trio_to_dpa(sri2dpa)
    canton_aliases, trio_codes = load_overrides()
    resolve = make_resolver(sri2dpa, name2dpa, trio2dpa, canton_prefix,
                            canton_aliases, trio_codes)

    # Resolve each DISTINCT trio once, then map back (much fewer trios than rows).
    trios = raw[["prov", "canton", "parroquia"]].drop_duplicates()
    resolved = trios.apply(lambda r: resolve(r["prov"], r["canton"], r["parroquia"]),
                           axis=1, result_type="expand")
    trios = trios.assign(cod_parr=resolved[0], metodo=resolved[1])
    raw = raw.merge(trios, on=["prov", "canton", "parroquia"], how="left")

    # Collapse SRI granularity onto the DPA parish: group then sum.
    keys = ["anio", "mes", "cod_ciuu4", "cod_parr", "tipo"]
    out = (raw.groupby(keys, dropna=False, as_index=False)["total"].sum()
              .sort_values(keys, ignore_index=True))

    out.to_parquet(OUT_PARQUET, index=False)

    # ---- audit + report ----
    audit = (raw.groupby(["prov", "canton", "parroquia", "cod_parr", "metodo"],
                         dropna=False, as_index=False)["total"].sum()
                .sort_values(["metodo", "total"], ascending=[True, False], ignore_index=True))
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")

    firms = raw["total"].sum()
    print("\nResolution by method (firm-records):")
    for method, g in raw.groupby("metodo")["total"].sum().sort_values(ascending=False).items():
        print(f"  {method:11} {g:>12,}  ({100 * g / firms:6.3f}%)")
    unresolved = raw[raw["cod_parr"].isna()]
    print(f"\nUnresolved firm-records: {unresolved['total'].sum():,} "
          f"in {unresolved[['prov','canton','parroquia']].drop_duplicates().shape[0]} trios")
    print(f"\nWrote {OUT_PARQUET}  ({len(out):,} rows)")
    print(f"Wrote {AUDIT_CSV}  ({len(audit):,} distinct trios)")
    return out


if __name__ == "__main__":
    main()
