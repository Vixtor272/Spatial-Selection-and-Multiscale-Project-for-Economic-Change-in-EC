#!/usr/bin/env python3
"""
other_taxpayer_types.py
=======================
Why the thesis studies private companies only.

The SRI data describe three kinds of taxpayer: private companies, natural
persons and public-sector entities. This script runs the SAME decomposition,
built the SAME way, on each of them and on the three pooled together, and
prints the numbers that show they are not the same kind of unit:

  * how many registered units each type has and how much they sell;
  * the national terms of the population path (total, selection, the part of
    selection between provinces, transmission).

The row for private companies reproduces the thesis headline exactly, which
is the check that "built the same way" is true.

Construction, identical for every type (it is the thesis panel's):
  firms   registry of active taxpayers of that type, by parish x CIIU section
          x month, kept only in the cells where that type declared sales;
  sales   V_VTX of that type, rows with blank sales dropped, deflated;
  then    ecuador_price/decomposition.py, yearly, survivors, $1,000 floor.

Which registry rows are which type:
  private companies   DESCRIPCION_SUBTIPO = "SECTOR PRIVADO"   <-> tipo SDD_NO_PUB
  natural persons     DESCRIPCION_TIPO    = "PERSONAS NATURALES" <-> tipo PN
  public sector       DESCRIPCION_SUBTIPO = "SECTOR PUBLICO"   <-> tipo SDD_PUB

A second table measures what pooling does. In the pooled panel it reports, for
each type, its share of the units and of the sales; and across parishes, how
closely the pooled fitness (growth of the number of units) follows the fitness
of natural persons and of private companies. If one type supplies nearly all
the units, the pooled decomposition is that type's decomposition.

Output -> results/other_taxpayer_types/summary.csv
          results/other_taxpayer_types/pooled_composition.csv

Run (reads the raw files, takes a few minutes):
    python -m ecuador_price.analysis.other_taxpayer_types
"""
import glob
import os

import pandas as pd

from ecuador_price import decomposition as dec
from ecuador_price import paths
from ecuador_price.build_db import build_active_firms as registry
from ecuador_price.build_db import map_parroquias_sri_to_dpa as crosswalk

OUT = paths.RESULTS / "other_taxpayer_types"
FREQ = "yearly"
WINDOWS = [(2017, 2023), (2019, 2020)]
KEYS = ["anio", "mes", "section", "cod_parr"]          # the grain of the thesis panel

TYPES = {            # label -> (sales `tipo`, how to pick the registry rows)
    "private companies": ("SDD_NO_PUB", lambda r: r["DESCRIPCION_SUBTIPO"] == "SECTOR PRIVADO"),
    "natural persons":   ("PN",         lambda r: r["DESCRIPCION_TIPO"] == "PERSONAS NATURALES"),
    "public sector":     ("SDD_PUB",    lambda r: r["DESCRIPCION_SUBTIPO"] == "SECTOR PUBLICO"),
}


def load_registry():
    """Every taxpayer type of the active-taxpayer files, with a DPA parish code:
    one row per (year, month, CIIU section, type, subtype, parish)."""
    by = ["DFC_ANIO_PK", "DFC_DESCRIPCION_MES", "CODIGO_OPERA_FAMILIA", "DESCRIPCION_TIPO",
          "DESCRIPCION_SUBTIPO", "DESCRIPCION_PROVINCIA", "DESCRIPCION_CANTON", "DESCRIPCION_PARROQUIA"]
    frames = []
    for path in sorted(glob.glob(os.path.join(registry.RAW_DIR, "sri_activos_*.csv"))):
        one = pd.read_csv(path, sep=registry.sniff_sep(path), dtype=str, encoding="latin-1",
                          keep_default_na=False)
        one["TOTAL"] = one["TOTAL"].astype(int)
        frames.append(one.groupby(by, as_index=False)["TOTAL"].sum())
    reg = pd.concat(frames, ignore_index=True)

    # names -> DPA parish code, with the same resolver the thesis build uses
    sri2dpa, name2dpa = registry.load_sri_to_dpa()
    trio2dpa, canton_prefix = registry.load_trio_to_dpa(sri2dpa)
    canton_aliases, trio_codes = registry.load_overrides()
    resolve = registry.make_resolver(sri2dpa, name2dpa, trio2dpa, canton_prefix,
                                     canton_aliases, trio_codes)
    names = ["DESCRIPCION_PROVINCIA", "DESCRIPCION_CANTON", "DESCRIPCION_PARROQUIA"]
    trios = reg[names].drop_duplicates()
    trios["cod_parr"] = [resolve(p, c, q)[0] for p, c, q in trios.itertuples(index=False)]
    reg = reg.merge(trios, on=names).dropna(subset=["cod_parr"])

    reg["anio"] = reg["DFC_ANIO_PK"].str.strip()
    reg["mes"] = reg["DFC_DESCRIPCION_MES"].str.extract(r"(\d{2})", expand=False)
    reg["section"] = reg["CODIGO_OPERA_FAMILIA"].str.strip()
    return reg


def load_sales():
    """The sales extract with a DPA parish code, rows with blank sales dropped:
    one row per (year, month, CIIU section, parish, tipo)."""
    sales = pd.read_csv(paths.SRI_SALES_EXTRACT, sep="\t", dtype={"parroquia": str},
                        usecols=["parroquia", "anio_fiscal", "mes_fiscal", "tipo", "ciiun4", "V_VTX"])
    sales = sales[sales["V_VTX"].notna()]
    sales = crosswalk.apply_crosswalk(sales, crosswalk.build_crosswalk(paths.DPA_XLSX))
    sales = sales.dropna(subset=["codigo_dpa"])
    sales["anio"] = sales["anio_fiscal"].astype(str)
    sales["mes"] = sales["mes_fiscal"].astype(str).str.zfill(2)
    sales["section"] = sales["ciiun4"].str[0]
    return (sales.groupby(KEYS[:3] + ["codigo_dpa", "tipo"], as_index=False)["V_VTX"].sum()
                 .rename(columns={"codigo_dpa": "cod_parr", "V_VTX": "total_sales"}))


def panel_for(firms, sales):
    """The thesis panel for one population: cells with BOTH a firm count and
    declared sales, with real sales."""
    panel = sales.merge(firms, on=KEYS, how="inner")
    ipc = pd.read_csv(paths.IPC_CSV)
    ipc.columns = ["year", "factor"]
    panel["year"] = panel["anio"].astype(int)
    panel["real_sales"] = panel["total_sales"] * panel["year"].map(dict(zip(ipc["year"], ipc["factor"])))
    return panel


def national_terms(label, panel):
    """Population-path terms of each window, plus the size of the population."""
    monthly = dec.monthly_table(panel)
    pp = dec.parish_period_table(monthly, FREQ)
    valid = dec.annual_valid_cells(monthly)
    rows = []
    for y0, y1 in WINDOWS:
        results = dec.run_window(pp, valid, f"{y0}_to_{y1}", (y0, 1), (y1, 1), FREQ)
        pop = next(r for r in dec.national_summary_rows(results, "w", (y0, 1), (y1, 1), "x", FREQ)
                   if r["path"] == "pop")
        survivors = (dec.valid_parishes(pp, str(y0), valid) & dec.valid_parishes(pp, str(y1), valid))
        start = pp[(pp["period"] == str(y0)) & pp["cod_parr"].isin(survivors)]
        rows.append({
            "population": label, "window": f"{y0}_to_{y1}", "parishes": len(survivors),
            "registered_units_start": start["firms"].sum(),
            "real_sales_start_bn": start["sales"].sum() / 1e9,
            "sales_per_unit_start": start["sales"].sum() / start["firms"].sum(),
            "total": pop["delta_trait_per_year"], "selection": pop["selection_total"],
            "sel_between_provinces": pop["sel_between_provinces"], "transmission": pop["transmission"]})
    return rows


def annual_by_parish(panel):
    """Units (mean monthly stock) and real sales of each parish in each year."""
    return dec.parish_period_table(dec.monthly_table(panel), FREQ)


def weighted_correlation(x, y, weight):
    """Correlation of x and y across parishes, each parish weighted by ``weight``."""
    weight = weight / weight.sum()
    dx, dy = x - (weight * x).sum(), y - (weight * y).sum()
    return (weight * dx * dy).sum() / ((weight * dx ** 2).sum() * (weight * dy ** 2).sum()) ** 0.5


def pooled_composition(own_panels, pooled_firms, pooled_sales, y0=2017, y1=2023):
    """What the pooled panel is made of, and whose growth its fitness follows.

    Works on the parishes of the pooled window y0 -> y1. A type's units and sales
    are counted in the cells of the pooled panel (a cell = parish x section x
    month where the three types together have both units and declared sales)."""
    pooled_panel = panel_for(pooled_firms, pooled_sales)
    pooled = annual_by_parish(pooled_panel)
    monthly = dec.monthly_table(pooled_panel)
    valid = dec.annual_valid_cells(monthly)
    parishes = sorted(dec.valid_parishes(pooled, str(y0), valid) & dec.valid_parishes(pooled, str(y1), valid))
    cells = pooled_panel[KEYS].drop_duplicates()

    def at(table, year, column):
        return table[table["period"] == str(year)].set_index("cod_parr")[column].reindex(parishes).fillna(0.0)

    pooled_units, pooled_units_end = at(pooled, y0, "firms"), at(pooled, y1, "firms")
    pooled_fitness = pooled_units_end / pooled_units
    rows = []
    for label, (firms, sales) in own_panels.items():
        # this type's units and sales inside the cells of the pooled panel
        inside = cells.merge(firms, on=KEYS, how="left").merge(sales, on=KEYS, how="left").fillna(
            {"total_firms": 0, "total_sales": 0.0})
        ipc = pd.read_csv(paths.IPC_CSV)
        ipc.columns = ["year", "factor"]
        inside["year"] = inside["anio"].astype(int)
        inside["real_sales"] = inside["total_sales"] * inside["year"].map(dict(zip(ipc["year"], ipc["factor"])))
        own = annual_by_parish(inside)
        units, units_end, own_sales = at(own, y0, "firms"), at(own, y1, "firms"), at(own, y0, "sales")
        both = (units > 0) & (units_end > 0)                 # parishes where this type is present at both ends
        fitness = (units_end / units)[both]
        rows.append({
            "population": label,
            "units": units.sum(), "share_of_units": units.sum() / pooled_units.sum(),
            "real_sales_bn": own_sales.sum() / 1e9, "share_of_sales": own_sales.sum() / at(pooled, y0, "sales").sum(),
            "sales_per_unit": own_sales.sum() / units.sum(),
            "units_growth_factor": units_end.sum() / units.sum(),
            "parishes_present": int(both.sum()),
            "parishes_where_majority_of_units": int((units / pooled_units > 0.5).sum()),
            "median_parish_share_of_units": (units / pooled_units).median(),
            "share_of_units_in_quito_guayaquil": units.reindex(["170150", "090150"]).sum() / units.sum(),
            "fitness_correlation_with_pooled": weighted_correlation(
                fitness, pooled_fitness[both], pooled_units[both]),
        })
    table = pd.DataFrame(rows)
    table.attrs["parishes"] = len(parishes)
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg, sales = load_registry(), load_sales()

    rows, firm_tables, sales_tables, own_panels = [], [], [], {}
    for label, (tipo, pick) in TYPES.items():
        firms = (reg[pick(reg)].groupby(KEYS, as_index=False)["TOTAL"].sum()
                 .rename(columns={"TOTAL": "total_firms"}))
        own_sales = sales[sales["tipo"] == tipo].drop(columns="tipo")
        firm_tables.append(firms)
        sales_tables.append(own_sales)
        own_panels[label] = (firms, own_sales)
        print(f"\n### {label}")
        rows += national_terms(label, panel_for(firms, own_sales))

    # the three pooled: what "all taxpayers" would measure
    print("\n### all three pooled")
    pooled_firms = pd.concat(firm_tables).groupby(KEYS, as_index=False)["total_firms"].sum()
    pooled_sales = pd.concat(sales_tables).groupby(KEYS, as_index=False)["total_sales"].sum()
    rows += national_terms("all three pooled", panel_for(pooled_firms, pooled_sales))

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print("\n=== population path, per year ===")
        print(table.to_string(index=False))
    print(f"\nsaved -> {OUT / 'summary.csv'}")

    composition = pooled_composition(own_panels, pooled_firms, pooled_sales)
    composition.to_csv(OUT / "pooled_composition.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print(f"\n=== what the pooled panel is made of, 2017 ({composition.attrs['parishes']} parishes) ===")
        print(composition.T.to_string(header=False))
    print(f"\nsaved -> {OUT / 'pooled_composition.csv'}")


if __name__ == "__main__":
    main()
