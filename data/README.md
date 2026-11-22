# Data

The data is **not included** in this repository. This page states, for every
number in the thesis, where the data came from and what was done to it:

1. [what to obtain](#raw-what-to-obtain), and from where;
2. [coverage and known gaps](#coverage-and-known-gaps);
3. [the whole chain](#the-whole-chain-from-the-raw-files-to-the-results), from the raw
   files to the table each analysis runs on;
4. [what the build writes](#processed-what-the-build-writes).

Paths are the ones named in [`ecuador_price/paths.py`](../ecuador_price/paths.py).

```
data/
  geo_overrides.csv     in the repository (hand-edited corrections, see below)
  raw/                  files as received: put them here, with these names
  processed/            written by ecuador_price/build_db: do not edit by hand
```

## `raw/`: what to obtain

The project crosses **two** SRI datasets: sales declarations (on request) and the
count of active taxpayers (public). Everything else is a reference table: codes,
names, a deflator and map shapes.

| file or folder | what it is | source |
| --- | --- | --- |
| `SRI_2015-2025_respaldo/BDD_AGR_VENTAS_PASTUDILLO_TR1170120260413045.csv` and `NOTAS.txt` | Aggregated sales declarations: one row per SRI parish × year × month × taxpayer type × 4-digit CIIU activity. Tab-separated, 11.8 million rows, 2015–2025, declarations registered up to 24 April 2026. | **On request.** Special request to the Servicio de Rentas Internas (SRI), handled through the USFQ Datahub. Not redistributed. Annex A of the Spanish thesis reproduces the variable dictionary and sample rows so the request can be repeated. |
| `Estadisticas de Contribuyentes Activos/sri_activos_<year>.csv` (2017–2025) | Number of taxpayers whose registration (RUC) is in the *active* state, by month, CIIU section, taxpayer type and subtype, province, canton and parish (names, not codes). It is a stock net of suspensions and cancellations, not the number of taxpayers ever registered: the SRI publishes registrations ("Estadísticas de inscripciones") and closures ("Estadísticas de cierres") as separate statistics on the same page, and in these files the stock of a parish × section cell falls from one month to the next 20,108 times. "Active" is an administrative state of the RUC; it does not mean the taxpayer sold or filed that month. The SRI's dictionary for these statistics (`SRI_Diccionario_Catastro.xlsx`, DD-05-2021-V01) describes the columns but does not define the state. One file per year, latin-1 text; the 2017–2020 files are separated by `;`, the later ones by `\|`. | **Public, downloaded directly.** SRI open data catalogue, <https://www.sri.gob.ec/datasets>; the yearly files are served as `https://descargas.sri.gob.ec/download/datosAbiertos/sri_activos_<year>.csv`. |
| `IPC_Ecuador.csv` | Yearly factor that turns each year's dollars into constant 2025 dollars. Two columns, `año,factor`. | Built by hand from the national consumer price index of INEC, <https://www.ecuadorencifras.gob.ec/indice-de-precios-al-consumidor/>. The values are in the table below. |
| `dpa.xlsx` | Parish codes: the SRI list (7 digits) and the official DPA list of INEC (6 digits, province + canton + parish), with names. | SRI / INEC catalogues. |
| `dic_ciiu.xlsx` | Dictionary of CIIU rev. 4 activity codes. | SRI / INEC catalogues. |
| `Anexos y Tablas para Municipios.xlsx` | Sheet "Mapeo Parroquias": province, canton and parish names → SRI parish code. Used to give codes to the active-taxpayer statistics. | Public, downloaded directly from the SRI document library (sri.gob.ec). |
| `manual_puntos_atencion_anexo.xlsx` | Sheet "BASE": SRI parish code → DPA parish code. Second half of the same match. | Public, downloaded directly from the Superintendencia de Bancos (superbancos.gob.ec, annex of its service-points manual). |
| `ORGANIZACION_TERRITORIAL_DEL_ESTADO_{PROVINCIAL,CANTONAL,PARROQUIAL}/` | Shapefiles of provinces, cantons and parishes (used only for maps and videos). Galápagos is not in them. | Official territorial organization of Ecuador (CONALI / INEC). |

For `dpa.xlsx`, `dic_ciiu.xlsx` and the shapefiles the exact download page was not
recorded when they were obtained; they are standard public reference files of the
institutions named.

Deflation factors in `IPC_Ecuador.csv`:

| year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factor | 1.103 | 1.084 | 1.080 | 1.082 | 1.079 | 1.083 | 1.082 | 1.045 | 1.023 | 1.007 | 1.000 |

## Coverage and known gaps

Checked file by file (last check: 18 November 2026, including a fresh download of
the active-taxpayer files made that day).

| dataset | years | complete? |
| --- | --- | --- |
| Sales extract | 2015–2025 | All 12 months in every year. |
| Active taxpayers | 2017–2023 | All 12 months in every year. |
| Active taxpayers | 2024 | **November is missing** in the file the SRI publishes (January–October, then December). The file on the SRI server was last changed on 16 January 2025 and is byte-for-byte the one used here, so re-downloading does not fix it. |
| Active taxpayers | 2025 | **December is missing** (January–November). The copy downloaded on 18 November 2026 is identical and carries the date 1 December 2025, so the published file appears to have stopped there. |
| Active taxpayers | 2015–2016 | Not published: the series starts in 2017. |

Consequences, all already applied in the code and the thesis:

- The panel can only start in **2017**, because a row needs both sales and a firm count.
- The main analysis uses **2017–2023**, the years with twelve months in both datasets.
- Any comparison that touches 2024 uses the **same eleven months at both ends**
  (January–October and December); `ecuador_price/analysis/robustness_data_decisions.py`
  does this and shows why (one annual step changes from −0.0183 to −0.0271 when
  November is removed from both years).
- 2025 is in the built table with eleven months and is not used in the thesis.

### Variables of the sales extract (from the SRI's `NOTAS.txt`)

| variable | meaning |
| --- | --- |
| `parroquia` | SRI parish code (7 digits) |
| `anio_fiscal`, `mes_fiscal` | fiscal year and month of the declaration |
| `tipo` | `PN` natural person, `SDD_NO_PUB` company outside the public sector, `SDD_PUB` public company |
| `ciiun4` | CIIU activity code, 4 digits |
| `num_rango_desde`, `num_rango_hasta` | lowest and highest number of taxpayers in the cell |
| `V_VTX` | total sales and net exports |
| `V_X` | net exports |
| `C_CTM` | total purchases and net imports |
| `C_M` | net imports |

The extract also carries a column `count_id` that the SRI note does not document;
this project does not use it as a count. Location, activity and type are the ones
of the taxpayer's registration at the date of the query, for every year.

## The whole chain, from the raw files to the results

Every step below is a script in [`ecuador_price/build_db/`](../ecuador_price/build_db)
or a function of [`ecuador_price/decomposition.py`](../ecuador_price/decomposition.py);
the file named in each step is where to read the exact rule.

### A. Sales (`prepare_sales_extract.py`, then `build_firm_revenue.py`)

1. **Start:** the sales extract, 11,758,306 rows (natural persons 7.7 million,
   private companies 3.7 million, public companies 0.3 million).
2. **Keep companies that report sales.** Natural persons are dropped. Rows whose
   sales field `V_VTX` is blank are dropped: 38 % of the private-company rows
   (1,427,681 of 3,726,283) are cells where taxpayers exist but no sales were
   declared that month. `V_VTX` is never zero or negative. Result: 2,418,225 rows.
3. **Keep private companies** (`tipo = SDD_NO_PUB`): 2,298,602 rows.
4. **Sales = `V_VTX`**, total sales including net exports. Blank exports count as 0.
5. **Activity:** the 4-digit CIIU code is reduced to its section letter (`C110` → `C`),
   because the firm counts only exist by section.
6. **Parish:** the 7-digit SRI code is converted to the 6-digit DPA code of INEC with
   44 manual fixes (territorial reforms), a direct match on the last six digits, and
   the rule that the urban parishes of a canton are merged into the cantonal head
   parish. 1,358 SRI codes become 1,043 DPA parishes.
7. **Add up** to one row per parish × section × month.

### B. Firm counts (`build_active_firms.py`)

1. **Start:** the active-taxpayer statistics, one file per year.
2. **Keep private companies** (`DESCRIPCION_SUBTIPO = SECTOR PRIVADO`).
3. **Parish:** the files give province, canton and parish *names*, not codes. Each
   name trio is matched to a DPA code through the two SRI reference tables and the
   hand-edited `geo_overrides.csv`; every trio and the method used is written to
   `parroquia_crosswalk_audit.csv`. Rows that cannot be matched are left out.
4. **Add up** to one row per parish × section × month. The count is a **stock**:
   companies registered as active that month.

### C. Crossing the two (`build_firm_revenue.py`)

The sales rows (A) are joined to the firm counts (B) on
**year, month, CIIU section, DPA parish**. A sales row with no firm count is
removed and written, with the reason, to `firm_revenue_enriched_dropped.csv`:

| | rows | sales |
| --- | ---: | ---: |
| kept: the analysis table | 475,325 | $1,313 bn |
| dropped: 2015–2016, before the firm counts start | 93,707 | $240 bn |
| dropped: the two months missing from the firm counts | 11,008 | $37 bn |
| dropped: parish or cell with sales but no registered firm that month | 56,724 | $3.7 bn |

So the population is: **registered private companies, in the parish × section ×
month cells where at least one of them declared sales.** Registered firms in cells
with no declaration are not counted (3–4 % of the registry nationally).
[`missing_matches_report.md`](../ecuador_price/build_db/missing_matches_report.md)
has the full breakdown.

### D. What the decomposition runs on (`decomposition.py`)

From the analysis table, for the window being compared:

1. **Real sales:** sales × that year's deflation factor (constant 2025 dollars).
2. **One row per parish and period:** firms are added across sections within a
   month, then the months of the period are averaged (a stock); sales are added over
   sections and months (a flow).
3. **Valid parishes:** a positive firm count and more than $1,000 of real sales in
   the year, at both ends of the window. Parishes valid at only one end are left out.
4. **Hierarchy:** the DPA code nests parishes in cantons (first 4 digits) and cantons
   in provinces (first 2 digits).
5. These tables are handed to the engine of Kemp et al. Population = number of
   firms; trait = log of real sales (and, for the other two paths, the level of
   sales and the firm count).

### E. What the resilience analysis runs on (`notebooks/resilience_analysis.ipynb`, `metro_regular_economy_resilience.ipynb`)

The same population and the same sales variable as the decomposition: **private
companies, sales = `V_VTX`** (total sales including exports), constant 2025 dollars,
January 2017 – October 2024. The sample is the 174 parishes with at least 50
registered companies in 2019.

| | series, troughs, recovery, firm growth | activity mix: breadth (entropy), oil and mega cells, mix-predicted change |
| --- | --- | --- |
| table | the analysis table (C) | `df_ciuu_dpa.csv` (the step-A2 table with DPA codes), private companies only |
| activity detail | CIIU section (letter) | CIIU 4 digits |
| sales | `V_VTX` | `V_VTX` |
| firms | registered active companies | none: there is no firm count at 4-digit level |

- **Breadth** of a parish is the effective number of activities, the exponential of the
  entropy of each 4-digit activity's share of the parish's 2019 sales.
- **Mega cells** are the top 1 % of 2019 parish × activity cells by sales per taxpayer.
  The extract documents only a range of taxpayers per cell, so sales are divided by the
  top of the range (`num_rango_hasta`): a guaranteed minimum of sales per taxpayer.
- Nothing uses the extract's undocumented `count_id` column.
- The two tables agree: in 2019 the 4-digit table's sales by section are within 2 % of
  the analysis table's in every section (it also keeps the few cells with sales but no
  registered firm that month).

Why `V_VTX` and not sales without exports: the thesis measures the formal activity of
the firms located in a place, whatever the destination of the sale. Exports are 10–13 %
of private-company sales and far more in exporting provinces. The decomposition with
domestic sales only is in `results/robustness_data_decisions/summary.csv`
(`variante = ventas_internas`); selection changes by at most 0.0006 in any window.

## `processed/`: what the build writes

Run in this order (each step reads the output of the one before):

| step | command | writes |
| --- | --- | --- |
| 0 | `python -m ecuador_price.build_db.prepare_sales_extract` | `df_ciuu_total_sales_purchases.csv`: companies only, rows that report sales (2.4 million rows) |
| 1 | `python -m ecuador_price.build_db.build_active_firms` | `active_firms_dpa.parquet` (firm counts with DPA parish codes) and `parroquia_crosswalk_audit.csv` |
| 2 | `python -m ecuador_price.build_db.build_firm_revenue` | **`firm_revenue_enriched_clean.parquet`**, the table every analysis reads: one row per parish × CIIU section × month with sales and number of firms (private companies, complete rows only), and `firm_revenue_enriched_dropped.csv`, the rows left out and why |
| side | `python -m ecuador_price.build_db.map_parroquias_sri_to_dpa` | `df_ciuu_dpa.csv` (the step-0 table with DPA parish codes, 4-digit activities; read by the resilience notebooks) and `sri_to_dpa_crosswalk.csv` |

`python -m ecuador_price.build_db.analyze_missing_matches` explains the dropped rows;
its conclusions are written up in
[`ecuador_price/build_db/missing_matches_report.md`](../ecuador_price/build_db/missing_matches_report.md).

## `geo_overrides.csv`

A short hand-edited table read by `build_active_firms.py`: canton name aliases and
the province–canton–parish names that cannot be matched to a DPA code
automatically. It is the only data file in the repository.
