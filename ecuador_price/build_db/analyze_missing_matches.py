#!/usr/bin/env python3
"""
analyze_missing_matches.py
==========================
Explain the rows that build_firm_revenue.py dropped: revenue
cells that never found a firm count (total_firms missing), and where the CIIU
section dimension is involved.

Every dropped row is missing total_firms, so the real question is WHY the merge
key (anio, mes, cod_ciuu4, cod_parr) had no firm match. We answer it with one
hierarchical rule -- each row is blamed on the first key level that fails:

  1. year not covered    the firm files only start in 2017
  2. month not covered    a 2017-2025 month with no firm rows at all
  3. parish not in month  that parish has no firms registered that month
  4. section not in month that CIIU section has no firms that month
  5. cell gap             parish and section both exist that month, but not in
                          the same cell (revenue there, zero registered firms)

This script only reads; it prints the tables behind missing_matches_report.md.

Run:
  python -m ecuador_price.build_db.analyze_missing_matches
"""
import pandas as pd

from ecuador_price import paths

CLEAN_PARQUET = paths.FIRM_REVENUE_PARQUET
DROPPED_CSV   = paths.FIRM_REVENUE_DROPPED_CSV
FIRMS_PARQUET = paths.ACTIVE_FIRMS_PARQUET

# --------------------------------------------------------------------------- #
# Load the three pieces: kept rows, dropped rows, firm counts.
# --------------------------------------------------------------------------- #
clean   = pd.read_parquet(CLEAN_PARQUET)
dropped = pd.read_csv(DROPPED_CSV,
                      dtype={"anio": str, "mes": str, "cod_ciuu4": str, "cod_parr": str})
firms   = pd.read_parquet(FIRMS_PARQUET).dropna(subset=["cod_parr"])

grain = len(clean) + len(dropped)
print(f"revenue grain rows : {grain:,}")
print(f"matched (clean)    : {len(clean):,} ({len(clean)/grain:.2%})")
print(f"dropped (no count) : {len(dropped):,} ({len(dropped)/grain:.2%})")
print(f"sales in dropped   : {dropped['total_sales'].sum():,.0f} "
      f"({dropped['total_sales'].sum()/(clean['total_sales'].sum()+dropped['total_sales'].sum()):.1%} of all sales)")
print(f"exports in dropped : {dropped['exports'].sum():,.0f}")

# --------------------------------------------------------------------------- #
# What the firm table actually covers (the reference sets for the merge keys).
# --------------------------------------------------------------------------- #
firm_years   = set(firms["anio"].unique())
firm_months  = set(map(tuple, firms[["anio", "mes"]].drop_duplicates().values))
parish_in    = firms.groupby(["anio", "mes"])["cod_parr"].agg(set).to_dict()
section_in    = firms.groupby(["anio", "mes"])["cod_ciuu4"].agg(set).to_dict()

def blame(r):
    a, m = r.anio, r.mes
    if a not in firm_years:               return "1. year not covered (2015-2016)"
    if (a, m) not in firm_months:         return "2. month not covered"
    if r.cod_parr not in parish_in[(a, m)]:  return "3. parish not in firm data that month"
    if r.cod_ciuu4 not in section_in[(a, m)]: return "4. section not in firm data that month"
    return "5. cell gap (parish & section present, not together)"

dropped["drop_reason"] = [blame(r) for r in dropped.itertuples(index=False)]

print("\n=== WHY total_firms IS MISSING ===")
summary = (dropped.groupby("drop_reason")
                  .agg(rows=("total_sales", "size"), sales=("total_sales", "sum")))
summary["share_rows"] = summary["rows"] / summary["rows"].sum()
print(summary.sort_index().to_string(
    formatters={"rows": "{:,}".format, "sales": "{:,.0f}".format,
                "share_rows": "{:.1%}".format}))

print("\nmonths present in revenue (2017-2025) but missing from firm data:")
rev_months = set(map(tuple, pd.concat([clean, dropped])[["anio", "mes"]].drop_duplicates().values))
print("  ", sorted(ym for ym in rev_months if ym[0] in firm_years and ym not in firm_months))

# --------------------------------------------------------------------------- #
# CIIU section (cod_ciuu4) view.
# --------------------------------------------------------------------------- #
print("\n=== CIIU SECTION (cod_ciuu4) ===")
rev_secs, firm_secs = set(clean["cod_ciuu4"]) | set(dropped["cod_ciuu4"]), set(firms["cod_ciuu4"])
print("in revenue but not firms:", sorted(rev_secs - firm_secs))
print("in firms but not revenue:", sorted(firm_secs - rev_secs))

allg = pd.concat([clean.assign(ok=1), dropped.assign(ok=0)], ignore_index=True)
win  = allg[allg["anio"].astype(int).between(2017, 2025)]
per  = win.groupby("cod_ciuu4").agg(grain=("ok", "size"), matched=("ok", "sum"))
per["dropped"]   = per["grain"] - per["matched"]
per["drop_rate"] = per["dropped"] / per["grain"]
print("\nper-section drop rate within 2017-2025 (sorted):")
print(per.sort_values("drop_rate", ascending=False).to_string(
    formatters={"grain": "{:,}".format, "matched": "{:,}".format,
                "dropped": "{:,}".format, "drop_rate": "{:.1%}".format}))

# --------------------------------------------------------------------------- #
# Reverse direction: firm cells (2017-2025) that never see any revenue.
# --------------------------------------------------------------------------- #
rev_keys = set(map(tuple, allg[["anio", "mes", "cod_ciuu4", "cod_parr"]].drop_duplicates().values))
fcells = firms[firms["anio"].isin(firm_years)][["anio", "mes", "cod_ciuu4", "cod_parr"]].drop_duplicates()
has_rev = [tuple(x) in rev_keys for x in fcells.values]
print(f"\n=== REVERSE: firm cells (2017-2025) with NO revenue ===")
print(f"firm cells: {len(fcells):,} | no revenue: {len(fcells) - sum(has_rev):,} "
      f"({1 - sum(has_rev)/len(fcells):.1%})")
