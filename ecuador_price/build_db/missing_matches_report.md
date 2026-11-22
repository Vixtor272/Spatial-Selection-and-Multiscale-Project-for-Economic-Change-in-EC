# Missing-match report — `firm_revenue_enriched`

Why **161,439 of 636,764** revenue cells (25.35%) get no firm count, and what
role the CIIU section dimension plays. Built from `firm_revenue_enriched_clean.parquet`
(kept) and `firm_revenue_enriched_dropped.csv` (dropped) against
`active_firms_dpa.parquet`. Reproduce with `analyze_missing_matches.py`.

Every dropped row is missing the **same** variable — `total_firms` — so the only
real question is why the merge key `(anio, mes, cod_ciuu4, cod_parr, tipo)` found
no firm row. `total_sales` and `exports` are never missing (a blank field is
summed as 0), and `tipo` is constant, so they never cause a drop.

---

## Headline

| | rows | share | sales (USD) |
|---|---:|---:|---:|
| matched (kept) | 475,325 | 74.65% | 1,313,225,592,533 |
| **unmatched (dropped)** | **161,439** | **25.35%** | **280,313,685,853** |

The dropped rows hold **17.6% of all sales** and **$29.7 B of exports** — but, as
the next table shows, almost all of that value is *structural* (years/months the
firm registry simply does not cover), not genuine cell-level mismatch.

---

## Why `total_firms` is missing

Each dropped row is blamed on the **first** merge-key level that fails:

| Reason | rows | share of drops | sales (USD) |
|---|---:|---:|---:|
| **1. Year not covered (2015–2016)** | 93,707 | 58.0% | 239,685,964,791 |
| **2. Month not covered** | 11,008 | 6.8% | 36,928,919,232 |
| **3. Parish not in firm data that month** | 7,351 | 4.6% | 1,295,922,408 |
| **4. Section not in firm data that month** | 0 | 0.0% | 0 |
| **5. Cell gap** (parish & section present, not together) | 49,373 | 30.6% | 2,402,879,423 |

Two facts jump out:

- **~99% of the *unmatched sales value* is reasons 1 + 2** ($276.6 B of $280.3 B).
  These are pure data-availability gaps, not analytical mismatch.
- **The fine-grained gaps (3 + 5) are many rows but tiny value** — 56,724 rows
  worth only **$3.7 B (1.3% of dropped sales)**. They are low-revenue cells.

### 1. Year not covered — 93,707 rows (58%)
The SRI active-firm files begin in **2017**, but revenue runs from **2015**. Every
2015–2016 cell therefore has revenue and no possible firm count. Not fixable
without earlier firm files; to work only with complete cases, set `YEAR_MIN = 2017`
in `build_firm_revenue.py`.

### 2. Month not covered — 11,008 rows (6.8%)
Within 2017–2025, exactly **two months are absent from the firm data**:

> **2024-11** and **2025-12**

Every other month (2017-01 … 2025-11, minus 2024-11) is present. These look like
missing/in-progress monthly source files. Adding `sri_activos` for those two
months would recover all 11,008 rows (and $36.9 B of sales). Worth confirming the
2025 file isn't simply still partial.

### 3. Parish not in firm data that month — 7,351 rows (4.6%)
The parish reported revenue but had **no active firm of any sector** registered
that month. Small in value ($1.3 B).

### 4. Section not in firm data that month — 0 rows
**No revenue CIIU section is ever wholly missing from a month.** Whenever the firm
data covers a month, every section that appears in revenue also appears somewhere
in the firm data that month. So the CIIU dimension never causes a *section-level*
miss — see the next section.

### 5. Cell gap — 49,373 rows (30.6%)
The parish **and** the section both exist in the firm data that month, just not in
the *same* cell: that parish×section×month had sales but **zero registered active
firms**. This is the genuine fine-grained mismatch. It is numerous but
low-value ($2.4 B) and heavily concentrated in a few sectors (below).

---

## CIIU section (`cod_ciuu4`) view

The merge uses the **section letter** (first character of `ciiun4`). The section
sets line up almost completely:

- **In revenue but not in firms:** *(none)* — every revenue section exists in the
  firm data, so no section is globally unmatchable.
- **In firms but not in revenue:** `NO TIENE` (no CIIU assigned) and `W`. These
  firm rows can never match revenue and feed the reverse gap (last section).
- **Oddities, both sides, and they match fine:** `'9'` is entirely the placeholder
  code `ciiun4 = '9999'` (214 rows; only 1.2% drop), and `'V'` (a non-standard
  section) drops just 3.9%. The junk codes are *not* the problem.

The real CIIU signal is how **unevenly** the drop rate falls across sections
(within 2017–2025, so the year gap doesn't distort it):

| section | grain | matched | dropped | drop rate |
|---|---:|---:|---:|---:|
| **K** financial & insurance | 17,959 | 9,088 | 8,871 | **49.4%** |
| **I** accommodation & food | 25,021 | 16,435 | 8,586 | **34.3%** |
| **U** extraterritorial orgs | 1,265 | 904 | 361 | 28.5% |
| C manufacturing | 42,509 | 33,656 | 8,853 | 20.8% |
| A agriculture | 52,244 | 42,848 | 9,396 | 18.0% |
| N admin & support | 33,486 | 27,498 | 5,988 | 17.9% |
| D electricity/gas | 6,676 | 5,914 | 762 | 11.4% |
| G wholesale & retail | 41,173 | 36,547 | 4,626 | 11.2% |
| B mining | 8,933 | 7,937 | 996 | 11.1% |
| L real estate | 15,074 | 13,583 | 1,491 | 9.9% |
| M professional/sci/tech | 23,041 | 21,236 | 1,805 | 7.8% |
| E water/waste | 39,814 | 36,943 | 2,871 | 7.2% |
| Q health & social | 23,760 | 22,211 | 1,549 | 6.5% |
| R arts & recreation | 12,428 | 11,637 | 791 | 6.4% |
| F construction | 36,646 | 34,467 | 2,179 | 5.9% |
| J information & comms | 16,668 | 15,756 | 912 | 5.5% |
| S other services | 56,789 | 53,869 | 2,920 | 5.1% |
| H transport & storage | 56,141 | 53,312 | 2,829 | 5.0% |
| P education | 20,583 | 19,557 | 1,026 | 5.0% |
| O public admin | 12,045 | 11,169 | 876 | 7.3% |
| T household activities | 327 | 297 | 30 | 9.2% |
| V *(non-standard)* | 311 | 299 | 12 | 3.9% |
| 9 *(`9999` placeholder)* | 164 | 162 | 2 | 1.2% |

**K (49%) and I (34%) dominate**, and they also dominate the cell-gap bucket
(K = 8,457, I = 7,634 of the 49,373 cell gaps). The likely reason is economic, not
a coding bug: financial/insurance and hospitality **revenue is reported where the
activity happens**, while the firm is registered at its fiscal domicile/HQ — so a
parish can book K or I sales in a month with no K/I firm on its own active-firm
roll. Sector-classification differences between the two SRI products (sales vs.
contribuyentes-activos) point the same way. Most other sections miss only ~5–11%.

---

## Reverse direction (context)

The mismatch runs both ways. Of the **781,180** firm cells in 2017–2025, only
**60.8%** have any matching revenue — **305,855 (39.2%)** have a registered firm
but **no reported sales** for that cell. The merge keeps the intersection of two
partially-overlapping registries; `total_firms` missing on the revenue side and
revenue missing on the firm side are the two faces of the same gap.

---

## What to do

1. **2015–2016 (58% of drops, 85% of dropped sales):** no firm data exists.
   Restrict firm-based analysis to **2017–2025** (`YEAR_MIN = 2017`).
2. **2024-11 and 2025-12 (6.8%, $36.9 B):** locate those two monthly
   `sri_activos` files (or confirm 2025 is still partial); rerun
   `build_active_firms.py` → `build_firm_revenue.py` to recover them.
3. **Parish & cell gaps (3 + 5, 35% of rows but ~1.3% of value):** genuine and
   expected — low-revenue cells, concentrated in **K** and **I** for
   domicile-vs-activity reasons. Treat as structural noise, or inspect K/I
   specifically if those sectors are central to the analysis.
