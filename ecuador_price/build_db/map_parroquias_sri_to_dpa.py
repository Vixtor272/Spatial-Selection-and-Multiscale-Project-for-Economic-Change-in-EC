#!/usr/bin/env python3
"""
map_parroquias_sri_to_dpa.py
============================
Builds a complete crosswalk from the SRI parroquia codification (7 digits:
REGION + PROVINCIA + CANTON + PARROQUIA) to the official INEC DPA
codification (6 digits: PROVINCIA + CANTON + PARROQUIA, the one used by
shapefiles), and applies it to the 'parroquia' column of the CIIU dataset.

Mapping rules, in order of precedence:
  1. MANUAL_FIXES ............ 44 codes affected by territorial reforms
                               (Santa Elena 2007, Santo Domingo 2007,
                               La Concordia 2013, cantonizations, etc.)
  2. direct .................. last 6 digits of the SRI code exist in the
                               DPA list (parroquia code >= 50)
  3. urbana_a_cabecera ....... urban parishes (parroquia code 01-49) are
                               collapsed into the cantonal head parish
                               (PPCC + '50'). This is where granularity
                               is intentionally lost.

Run (the defaults are the project files named in ecuador_price/paths.py):
  python -m ecuador_price.build_db.map_parroquias_sri_to_dpa

  reads   data/raw/dpa.xlsx, data/processed/df_ciuu_total_sales_purchases.csv
  writes  data/processed/df_ciuu_dpa.csv, data/processed/sri_to_dpa_crosswalk.csv

Any of the four files can be replaced on the command line:
  python -m ecuador_price.build_db.map_parroquias_sri_to_dpa \
      --xlsx my_dpa.xlsx --csv my_sales.csv \
      --out df_ciuu_dpa.csv --crosswalk-out sri_to_dpa_crosswalk.csv

IMPORTANT: 'codigo_dpa' is a zero-padded 6-char string (e.g. '010150').
When re-reading the output CSV, preserve leading zeros with:
    pd.read_csv(path, dtype={'codigo_dpa': str})
"""

import argparse
import openpyxl
import pandas as pd

from ecuador_price import paths

SHEET_XWALK = "Codigo Parroquias DPA"   # SRI list (cols A-B) + DPA list (cols C-D)
SHEET_BASE = "DPA_Base"                 # DPA with provincia/canton names

# SRI codes that cannot be resolved mechanically. Each is a consequence of a
# territorial reform that happened after the SRI registered the code, or a
# quirk of the SRI registry itself.
MANUAL_FIXES = {
    # --- Provincia de Santa Elena (creada 2007, antes Guayas 09-15/17/26) ---
    "1091750": "240150",  # SANTA ELENA (cabecera del nuevo canton 2401)
    "1091701": "240150",  # BALLENITA (parr. urbana de Santa Elena) -> cabecera
    "1091702": "240150",  # SANTA ELENA (parr. urbana) -> cabecera
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
    "1091515": "240351",  # ANCONCITO (hoy parroquia propia en la DPA)
    "1091552": "240352",  # JOSE LUIS TAMAYO (MUEY)
    # --- Provincia de Santo Domingo de los Tsachilas (creada 2007, antes Pichincha 17-06) ---
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
    # --- La Concordia (Esmeraldas 08-08 -> Santo Domingo 23-02, referendum 2013) ---
    "1080850": "230250",  # LA CONCORDIA
    "1080851": "230251",  # MONTERREY
    "1080852": "230252",  # LA VILLEGAS
    "1080853": "230253",  # PLAN PILOTO
    # --- Cantonizaciones / absorciones ---
    "1080151": "080650",  # ATACAMES: ex parroquia de Esmeraldas, canton propio desde 1991
    "1090151": "090150",  # CHONGON: absorbida por el area urbana de Guayaquil
    "1120751": "121350",  # QUINSALOMA: ex parroquia de Ventanas, canton propio desde 2007
    "3150755": "220453",  # SAN JOSE DE PAYAMINO: codigo viejo de Napo -> Loreto, Orellana (1998)
    "2011551": "010852",  # EL CARMEN DE PIJILI: el SRI la registra bajo Camilo Ponce
                          #   Enriquez (0115); la DPA la mantiene en Santa Isabel (010852)
    # --- Subdivisiones solo-SRI sin parroquia DPA propia -> cabecera cantonal ---
    "1131251": "131250",  # SOSOTE -> Rocafuerte (Manabi)
    "2010455": "010450",  # LA PAZ -> Nabon (Azuay)
    # --- Correcciones de datos ---
    "20610":   "061050",  # CUMANDA: codigo truncado en el catalogo SRI (= 2061050)
    "3220254": "220254",  # TIPUTINI (Aguarico, Orellana): omitida en la lista DPA de la
                          #   hoja 'Codigo Parroquias DPA' pero vigente en la DPA oficial
                          #   del INEC y presente en la hoja 'DPA_Base'
}
MANUAL_NAME_FALLBACK = {"220254": "TIPUTINI"}  # name for codes missing from the sheet-3 DPA list


def build_crosswalk(xlsx_path: str) -> pd.DataFrame:
    """Return one row per SRI code with its DPA equivalent."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)

    rows = list(wb[SHEET_XWALK].iter_rows(values_only=True))[1:]
    sri = [((r[0] or "").strip(), str(r[1]).strip()) for r in rows if r[1] is not None]
    dpa_names = {str(r[3]).strip(): (r[2] or "").strip() for r in rows if r[3] is not None}
    dpa_names.update(MANUAL_NAME_FALLBACK)

    # provincia / canton names per PPCC prefix, taken from DPA_Base
    ppcc_names = {}
    for r in list(wb[SHEET_BASE].iter_rows(values_only=True))[4:]:
        if r[1] and r[2] and r[4] and r[5]:
            key = f"{str(r[1]).zfill(2)}{str(r[2]).zfill(2)}"
            ppcc_names.setdefault(key, (str(r[4]).strip(), str(r[5]).strip()))
    # new-province cantons absent from DPA_Base
    ppcc_names.setdefault("2301", ("SANTO DOMINGO DE LOS TSACHILAS", "SANTO DOMINGO"))
    ppcc_names.setdefault("2302", ("SANTO DOMINGO DE LOS TSACHILAS", "LA CONCORDIA"))
    ppcc_names.setdefault("2401", ("SANTA ELENA", "SANTA ELENA"))
    ppcc_names.setdefault("2402", ("SANTA ELENA", "LA LIBERTAD"))
    ppcc_names.setdefault("2403", ("SANTA ELENA", "SALINAS"))

    out, unmatched = [], []
    for name, code in sri:
        if code in MANUAL_FIXES:
            target, method = MANUAL_FIXES[code], "manual"
        elif len(code) == 7 and code[1:] in dpa_names:
            target, method = code[1:], "direct"
        elif len(code) == 7 and int(code[5:7]) < 50 and code[1:5] + "50" in dpa_names:
            target, method = code[1:5] + "50", "urbana_a_cabecera"
        else:
            unmatched.append((code, name))
            continue
        prov, cant = ppcc_names.get(target[:4], ("", ""))
        out.append({
            "codigo_sri": code,
            "parroquia_sri": name,
            "codigo_dpa": target,
            "parroquia_dpa": dpa_names.get(target, ""),
            "provincia": prov,
            "canton": cant,
            "metodo": method,
        })
    if unmatched:
        raise ValueError(f"SRI codes left unmapped: {unmatched}")
    return pd.DataFrame(out)


def apply_crosswalk(df: pd.DataFrame, xwalk: pd.DataFrame,
                    col: str = "parroquia") -> pd.DataFrame:
    """Add codigo_dpa / parroquia_dpa columns next to the SRI `col`."""
    lookup = xwalk.set_index("codigo_sri")[["codigo_dpa", "parroquia_dpa"]]
    keys = df[col].astype(str).str.strip()
    df = df.copy()
    df["codigo_dpa"] = keys.map(lookup["codigo_dpa"])
    df["parroquia_dpa"] = keys.map(lookup["parroquia_dpa"])

    n_bad = df["codigo_dpa"].isna().sum()
    print(f"Rows mapped: {len(df) - n_bad}/{len(df)} "
          f"({(len(df) - n_bad) / len(df):.1%})")
    if n_bad:
        print("WARNING - SRI codes not found in the crosswalk:",
              sorted(keys[df["codigo_dpa"].isna()].unique()))
    return df


def aggregate_to_dpa(df: pd.DataFrame) -> pd.DataFrame:
    """Optional: collapse rows that now share the same DPA parroquia.

    Sums the value columns; keeps the widest contributor-count range.
    Use only if you no longer need the SRI-level detail.
    """
    keys = ["codigo_dpa", "parroquia_dpa", "anio_fiscal", "mes_fiscal",
            "tipo", "ciiun4"]
    agg = {"num_rango_desde": "min", "num_rango_hasta": "max",
           "count_id": "sum", "V_VTX": "sum", "V_X": "sum",
           "C_CTM": "sum", "C_M": "sum", "total_sales": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.groupby(keys, as_index=False).agg(agg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default=paths.DPA_XLSX)
    ap.add_argument("--csv", default=paths.SALES_CSV)
    ap.add_argument("--out", default=paths.SALES_DPA_CSV)
    ap.add_argument("--crosswalk-out", default=paths.SRI_TO_DPA_CROSSWALK_CSV)
    ap.add_argument("--aggregate", action="store_true",
                    help="also collapse rows to the DPA parroquia level")
    args = ap.parse_args()

    xwalk = build_crosswalk(args.xlsx)
    xwalk.to_csv(args.crosswalk_out, index=False, encoding="utf-8-sig")
    print(f"Crosswalk: {len(xwalk)} SRI codes -> "
          f"{xwalk['codigo_dpa'].nunique()} DPA parroquias "
          f"({dict(xwalk['metodo'].value_counts())})")
    df = pd.read_csv(args.csv, dtype={"parroquia": str})
    df = apply_crosswalk(df, xwalk)
    if args.aggregate:
        df = aggregate_to_dpa(df)
        print(f"Aggregated to {len(df)} DPA-level rows")
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()