"""Shared pieces for the tests: a small invented economy that needs no SRI data."""
import numpy as np
import pandas as pd
import pytest

# 3 provinces x 2 cantons x 3 parishes = 18 parishes, DPA-style 6-digit codes
# (2 digits province + 2 canton + 2 parish), observed every month of two years.
PARISH_CODES = [f"{prov:02d}{canton:02d}{parish:02d}"
                for prov in (1, 9, 17) for canton in (1, 2) for parish in (50, 51, 52)]
YEAR0, YEAR1 = 2020, 2021


def make_monthly(firm_growth=None, seed=7):
    """A table shaped like decomposition.monthly_table(): one row per
    (parish, year, month) with ``firms`` and ``sales``.

    firm_growth: None -> each parish gets its own random growth of the firm
                 count between the two years; a number -> every parish grows
                 by exactly that factor (proportional, "Gibrat" growth).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for code in PARISH_CODES:
        firms0 = rng.integers(5, 400)
        sales_per_firm0 = rng.uniform(2_000, 30_000)          # dollars per month
        g_firms = firm_growth if firm_growth is not None else rng.uniform(0.7, 1.6)
        g_sales_per_firm = rng.uniform(0.8, 1.3)
        for year, firms, spf in [(YEAR0, firms0, sales_per_firm0),
                                 (YEAR1, firms0 * g_firms, sales_per_firm0 * g_sales_per_firm)]:
            for month in range(1, 13):
                rows.append({"cod_parr": code, "year": year, "mes_i": month,
                             "firms": float(firms), "sales": float(firms * spf)})
    return pd.DataFrame(rows)


@pytest.fixture
def monthly():
    return make_monthly()
