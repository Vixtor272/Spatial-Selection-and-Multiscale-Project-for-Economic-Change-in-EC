"""The headline numbers of the thesis, recomputed from the SRI panel and
compared with the table stored in results/. Skipped when the data is not on
this machine (the SRI extract is not distributed with the repository)."""
import pandas as pd
import pytest

from ecuador_price import decomposition as dec
from ecuador_price import paths

STORED = paths.ROOT / "results" / "decomposition" / "yearly_consecutive_2017_2023" / "decomposition_summary.csv"
TERMS = ["delta_trait_per_year", "selection_total", "sel_between_provinces",
         "sel_canton_within_province", "sel_parish_within_canton", "transmission"]

needs_data = pytest.mark.skipif(
    not (paths.FIRM_REVENUE_PARQUET.exists() and STORED.exists()),
    reason="SRI panel or stored results not available")


@needs_data
def test_headline_2017_2023_reproduces_the_stored_table():
    monthly = dec.monthly_table(dec.load_real_panel())
    pp = dec.parish_period_table(monthly, "yearly")
    valid = dec.annual_valid_cells(monthly)
    results = dec.run_window(pp, valid, "2017_to_2023", (2017, 1), (2023, 1), "yearly")
    new = pd.DataFrame(dec.national_summary_rows(
        results, "2017_to_2023", (2017, 1), (2023, 1), "headline", "yearly")).set_index("path")

    stored = pd.read_csv(STORED)
    stored = stored[stored["window"] == "2017_to_2023"].set_index("path")
    for path in ["pop", "inc", "den"]:
        for term in TERMS:
            assert new.loc[path, term] == pytest.approx(stored.loc[path, term], abs=1e-12), (path, term)


@needs_data
def test_the_sign_pattern_the_thesis_reports():
    """Selection is negative and mostly between provinces; transmission is
    positive and larger (population path, 2017 -> 2023)."""
    stored = pd.read_csv(STORED)
    row = stored[(stored["window"] == "2017_to_2023") & (stored["path"] == "pop")].iloc[0]
    assert row["selection_total"] < 0 < row["transmission"]
    assert abs(row["transmission"]) > abs(row["selection_total"])
    assert row["share_between_provinces"] > 0.5
    assert abs(row["identity_residual"]) < 1e-12
