# The math of the two paths (`_pop` and `_inc`), in detail

This note derives, step by step, exactly what the decomposition engine
(`UrbanTraitSelection/calculation_scripts/runPrice.py`, imported unchanged by
`ecuador_price/decomposition.py`) computes for the two
"paths" that appear everywhere in the outputs — the **population path**
(`_pop`, superscript $p$ in the figures) and the **income path** (`_inc`,
superscript $i$) — and then lists what changes mathematically relative to the
Chicago data. Code references point to the functions in `runPrice.py` so every
formula can be checked against the implementation.

---

## 1. Setup and notation

Fix one **parent** unit $P$ (say, a canton) and its **children** $j = 1,\dots,J$
(its parishes), and one time window from the initial year to the final year,
$N$ years apart. Primes ($'$) denote final-year values. For each child:

| Symbol                            | Meaning (Ecuador)                                                             | CSV / code column                                             |
| --------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| $n_j,\; n_j'$                   | firm stock, initial and final                                                 | `PopInitial_*`, `PopFinal_*`                              |
| $w_j = n_j'/n_j$                | **fitness** (pop path): relative firm-count growth                      | `c_w_c` in `_part_a_initial_child_preparation`            |
| $z_j,\; z_j'$                   | **trait** (pop path): mean log real sales, $\mathbb{E}[\log y]$       | `AvgLogIncInitial_*`, `AvgLogIncFinal_*`                  |
| $y_j$                           | **trait level** (inc path): real sales level, $\mathbb{E}[y]$ initial | `ActualAvgIncInitial_*` $= e^{\texttt{LogAvgIncInitial}}$ |
| $g_j$                           | the child's own per-year growth (which growth depends on the path, see below) | `G_avg_log_inc_*` or `AvgG_emp_*` / `AvgG_inc_*`        |
| $p_j = \dfrac{n_j}{\sum_k n_k}$ | initial population share among siblings                                       | `c_p_c_P_grp` in `_part_b_parent_group_aggregates`        |

Two parent-level weighted means are precomputed (`_part_b`):

$$
\bar w_P = \sum_j p_j\, w_j, \qquad \bar z_P = \sum_j p_j\, z_j, \qquad \mu_P = \sum_j p_j\, y_j .
$$

Note the useful identity $\bar w_P = \sum_j \frac{n_j}{N_P}\frac{n_j'}{n_j} = \frac{N_P'}{N_P}$:
the mean fitness of the children **is** the parent's own population growth factor.

A second identity drives everything below. The **final** population share of
child $j$ is

$$
p_j' \;=\; \frac{n_j'}{N_P'} \;=\; \frac{n_j w_j}{N_P \bar w_P} \;=\; p_j\,\frac{w_j}{\bar w_P},
$$

so the *change in shares* is completely determined by relative fitness:

$$
p_j' - p_j \;=\; p_j\left(\frac{w_j}{\bar w_P} - 1\right). \tag{1}
$$

---

## 2. The population path (`_pop`): who moves, the firms

**Question it answers:** how much of the change in the (firm-weighted) average
prosperity comes from *firms redistributing themselves* across places, versus
places becoming more prosperous internally?

### 2.1 The quantity being decomposed

The parent's trait is the firm-weighted mean of its children's traits,
$\bar z_P = \sum_j p_j z_j$. Its change over the window, per year, is the
left-hand side the code calls `LHS_AvgG_pop`:

$$
\gamma^{p}_P \;=\; \frac{\bar z_P' - \bar z_P}{N} \;=\; \frac{1}{N}\Big(\sum_j p_j' z_j' - \sum_j p_j z_j\Big).
$$

### 2.2 Add-and-subtract: the exact split

Insert $\sum_j p_j' z_j$ (final shares, initial traits):

$$
\bar z_P' - \bar z_P
= \underbrace{\sum_j (p_j' - p_j)\, z_j}_{\text{shares moved, traits fixed}}
\;+\; \underbrace{\sum_j p_j'\,(z_j' - z_j)}_{\text{traits moved, at final shares}} . \tag{2}
$$

This is exact — no approximation. The first piece is **selection**, the second
is **transmission**.

### 2.3 The selection term is a covariance

Substitute the share-change identity (1) into the first piece:

$$
\sum_j (p_j'-p_j)\, z_j \;=\; \sum_j p_j\left(\frac{w_j}{\bar w_P}-1\right) z_j .
$$

Because the share changes sum to zero, $\sum_j (p_j'-p_j) = 0$, you may
subtract any constant from $z_j$ without changing the sum; subtract $\bar z_P$:

$$
\sum_j (p_j'-p_j)\, z_j \;=\; \sum_j p_j\left(\frac{w_j}{\bar w_P}-1\right)\left(z_j-\bar z_P\right)
\;=\; \operatorname{Cov}_p\!\left(\frac{w_j}{\bar w_P},\, z_j\right),
$$

where $\operatorname{Cov}_p$ is the covariance taken with the initial shares
$p_j$ as probability weights (note $\mathbb{E}_p[w_j/\bar w_P] = 1$ and
$\mathbb{E}_p[z_j] = \bar z_P$, so this is a textbook weighted covariance:
mean-deviation of one variable times mean-deviation of the other, weighted by
$p_j$). Dividing by $N$ to annualize gives **exactly** the code
(`_part_d_calculate_selection_terms`):

$$
\boxed{\;S^{pop}_P \;=\; \frac{1}{N}\sum_j p_j\left(\frac{w_j}{\bar w_P}-1\right)\left(z_j-\bar z_P\right)\;}
\qquad(\texttt{Sel\_P\_from\_child\_pop})
$$

**How to read it.** It is positive when children with above-average firm
growth ($w_j > \bar w_P$) are also the children with above-average initial log
sales ($z_j > \bar z_P$): firms pile into already-big markets, dragging the
mean up *without any market internally improving*. It is negative when firms
accumulate in initially small markets (catch-up / dispersion). Equivalently,
by the derivation, it is the per-year change of the mean trait you would see
**if every child's trait were frozen** and only the firm shares moved — pure
reshuffling.

### 2.4 The transmission term

The second piece of (2), using $p_j' = p_j w_j / \bar w_P$ and writing the
child's own per-year trait change $g_j = (z_j' - z_j)/N$:

$$
\boxed{\;T_P \;=\; \frac{1}{\bar w_P}\sum_j p_j\, w_j\, g_j\;}
\qquad(\texttt{TransmissionDirectChildGrowth\_P\_pop})
$$

— the **fitness-weighted average of the children's internal growth**. The
weighting matters: growth in a child that expanded ($w_j$ large) counts for
more, because by the final year that child holds more of the firms. Together,
(2) reads per year

$$
\gamma^p_P \;=\; S^{pop}_P + T_P ,
$$

which is the classic **Price equation**
$\Delta\bar z = \operatorname{Cov}(w,z)/\bar w + \mathbb{E}[w\,\Delta z]/\bar w$
with initial-share weighting. The engine verifies it unit-by-unit
(`SanChk_SingleLevel_pop_*`, and the plots in `sanity_check_*/`): residuals
are at machine precision.

### 2.5 Stacking levels: the multiscale identity

Now let the children themselves be parents one level down. Replace each
child's internal growth $g_j$ by its own Price split
$g_j \to \gamma^p_j = S^{pop}_j + T_j$ and push the fitness weights through.
Define the **transmission weights** used to roll any child-level quantity $x_j$
up to $P$ on the pop path (`_part_e_calculate_transmitted_terms`, `'_pop'`
branch):

$$
\phi_j \;=\; \frac{p_j\, w_j}{\bar w_P} \;=\; p_j' \qquad\Longrightarrow\qquad
\widetilde x_P = \sum_j \phi_j\, x_j
$$

(i.e., transmitted quantities are averaged with **final** firm shares).
Unrolling parish $\to$ canton $\to$ province $\to$ country gives the exact
four-term national identity used in all the summary figures:

$$
\gamma^{p}_{EC}
= \underbrace{S^{pop}_{EC}}_{\text{between provinces}}
+ \underbrace{\widetilde S^{pop}_{prov}}_{\substack{\text{between cantons}\\ \text{within provinces}}}
+ \underbrace{\widetilde S^{pop}_{can}}_{\substack{\text{between parishes}\\ \text{within cantons}}}
+ \underbrace{\widetilde T_{par}}_{\substack{\text{within-parish}\\ \text{growth}}}
$$

(`AvgG_pop_country = Sel_country_from_province_pop + Transmitted_Sel_province_to_country_pop + Transmitted_Sel_canton_to_country_pop + Transmitted_AvgG_parish_to_country_pop`,
checked by `SanChk_MultiLevel_pop_parish_to_country`). Statistically this is
the **law of total covariance**: the covariance of fitness and trait over all
parishes splits into the covariance of their group means at each nesting level
plus the within-group covariances, each rolled up with the appropriate weights.

---

## 3. The income path (`_inc`): who grows, the dollars

**Question it answers:** how much of aggregate sales growth comes from sales
growing faster *where sales are already large* (compounding of rich places),
versus the growth of the average place?

Here nothing needs to "move". The path re-derives the Price logic treating
**each dollar of sales as the replicating entity**, whose "fitness" is the
growth rate of the unit it sits in.

### 3.1 The quantity being decomposed

The firm-share-weighted mean sales level of the parent is
$\mu_P = \sum_j p_j y_j$, where $y_j = e^{\texttt{LogAvgIncInitial}_j}$ is the
child's initial sales level. If each child's sales grow at the per-year rate
$g_j$ (at the base level $g_j = \texttt{AvgG\_emp}_j = \Delta\log \mathbb{E}[y]_j / N$;
at higher levels the recursion uses $g_j = \texttt{AvgG\_inc}_j$), then to first
order the mean grows at

$$
\gamma^{i}_P \;=\; \frac{\sum_j p_j\, g_j\, y_j}{\mu_P}
\qquad(\texttt{AvgG\_inc\_P}),
$$

i.e., each child's growth rate weighted by *how many dollars* it applies to.

### 3.2 Add-and-subtract again: the covariance appears

Let $\bar g = \sum_j p_j g_j$ be the plain (firm-weighted) average of the
children's growth rates (`ExpectedChildEmpiricalGrowth_P_inc`). Then

$$
\gamma^i_P - \bar g
= \frac{\sum_j p_j\, g_j\, y_j}{\mu_P} - \sum_j p_j g_j
= \frac{1}{\mu_P}\sum_j p_j\, g_j\,\bigl(y_j - \mu_P\bigr)
= \frac{1}{\mu_P}\sum_j p_j\,\bigl(g_j-\bar g\bigr)\bigl(y_j - \mu_P\bigr),
$$

where the last step again uses that a constant ($\bar g$) times
$\sum_j p_j (y_j - \mu_P) = 0$ can be inserted for free. That is the code's
income selection term (`_part_d`):

$$
\boxed{\;S^{inc}_P \;=\; \frac{1}{\mu_P}\sum_j p_j\,(g_j - \bar g)\,(y_j - \mu_P)
\;=\; \frac{\operatorname{Cov}_p(g_j,\, y_j)}{\mu_P}\;}
\qquad(\texttt{Sel\_P\_from\_child\_inc})
$$

No $1/N$ here: $g_j$ is already a per-year rate, so the covariance is already
per year (contrast §2.3, where the covariance of a growth *factor* with a
trait had to be annualized).

**The cleanest way to read it.** Define income shares
$q_j = p_j y_j / \mu_P$ (child $j$'s share of the parent's sales). Then
$\gamma^i_P = \sum_j q_j g_j$ is the **dollar-weighted** mean growth and
$\bar g = \sum_j p_j g_j$ the **unit-weighted** mean growth, so

$$
S^{inc}_P \;=\; \underbrace{\textstyle\sum_j q_j g_j}_{\text{dollar-weighted}} - \underbrace{\textstyle\sum_j p_j g_j}_{\text{unit-weighted}} .
$$

Positive income selection $\iff$ dollars grow faster than units $\iff$ growth
is concentrated in already-rich children ($y_j$ big where $g_j$ big) — the
covariance form makes the mechanism explicit. Negative income selection is
exactly **$\beta$-convergence** among siblings: initially poorer children
growing faster in rate terms. The single-level identity checked by the engine
(`SanChk_SingleLevel_inc_*`) is

$$
\gamma^i_P = \bar g + S^{inc}_P .
$$

### 3.3 Transmission upward on the income path

Rolling child quantities up to higher levels uses **initial** shares,
$\phi_j = p_j$ (`_part_e`, `'_inc'` branch) — no fitness reweighting, because
on this path the "population" (the dollars) is reweighted inside the
covariance itself via $y_j$. The national identity has the same four-term
shape as the pop path with $S^{inc}$ and $\bar g$ in place of $S^{pop}$
and $T$.

---

## 4. The two paths side by side

|                                | **`_pop` (population path)**                                  | **`_inc` (income path)**                     |
| ------------------------------ | --------------------------------------------------------------------- | ---------------------------------------------------- |
| replicating entity             | a firm                                                                | a dollar of sales                                    |
| fitness                        | $w_j = n_j'/n_j$ (firm-count growth factor)                         | $g_j$ (the unit's own sales growth **rate**) |
| trait                          | $z_j = \mathbb{E}[\log y]$ (mean log sales)                         | $y_j = \mathbb{E}[y]$ (sales level)                |
| selection term                 | $\frac{1}{N}\operatorname{Cov}_p\!\big(\tfrac{w}{\bar w},\, z\big)$ | $\frac{1}{\mu}\operatorname{Cov}_p(g,\, y)$        |
| plain-language selection       | mean-trait change from firm reshuffling alone                         | dollar-weighted minus unit-weighted growth           |
| transmission weights$\phi_j$ | $p_j w_j/\bar w = p_j'$ (final shares)                              | $p_j$ (initial shares)                             |
| base-level growth$g_j$       | $\Delta \mathbb{E}[\log y]/N$ (`G_avg_log_inc`)                   | $\Delta \log\mathbb{E}[y]/N$ (`AvgG_emp`)        |
| selection$>0$ means          | firms accumulate where sales already high (sorting)                   | sales grow fastest where already high (divergence)   |
| selection$<0$ means          | firms accumulate in initially small markets                           | $\beta$-convergence among siblings                 |
| annualization                  | covariance$\div N$                                                  | none needed ($g$ already per-year)                 |

The two paths are **not** two estimates of the same thing — they decompose two
different aggregates ($\bar z$ vs $\mu$) along two different notions of
fitness, and can legitimately disagree in sign. In Ecuador 2017–2023 they
mostly agree: selection is negative at the province scale on both paths.

Derived metrics used in the plots, for any component $X$ of a unit whose total
growth is $\gamma$:

$$
\text{NS}(X) = 100\cdot\frac{X}{\gamma}
\quad(\text{signed; all components of a unit sum to }100\%),\qquad
\text{MSS}(X) = 100\cdot\frac{|X|}{\sum_k |X_k|}
\quad(\text{sign-blind local dominance}).
$$

NS$_{ec}$ divides by the **national** $\gamma$ instead of the unit's own
(Chicago's NS$_{st}$), making units comparable on a common yardstick.

---

## 5. What changes mathematically vs. the Chicago data

The engine (`runPrice.py`) is byte-identical — every formula above is executed
by the same code. Besides Ecuador having one level fewer
(parish→canton→province→country vs bg→tract→community→county→state), the
mathematical content of the *inputs* differs in these ways:

1. **The trait is built from totals, not from a within-unit distribution.**
   Chicago's base unit (block group) comes with a binned household-income
   distribution (ACS table B19001), so
   $z = \mathbb{E}[\log y] = \sum_b s_b \log(m_b)$ and
   $\log\mathbb{E}[y] = \log\big(\sum_b s_b m_b\big)$ (bin shares $s_b$,
   midpoints $m_b$) are **different numbers at the base level**, their gap
   being within-unit inequality (Jensen's inequality:
   $\mathbb{E}[\log y] \le \log \mathbb{E}[y]$). Ecuador's parish is observed
   only as a total: the code sets
   $z_{par} = \log(\text{total real sales}) = \texttt{LogAvgIncome}_{par}$, so
   **the two traits coincide at the base** and the Jensen gap only opens at
   aggregated levels (parent $z$ = firm-weighted mean of children's logs vs
   $\log$ of summed sales). Also note the trait is log **total** parish sales,
   not per-firm sales — it contains scale. The identities are unaffected (they
   hold for any trait), but interpretation shifts: pop-path selection covaries
   firm growth with initial *market size*, not with per-firm productivity.
   (The Gini figures deliberately correct to per-firm sales, $y_j/n_j$.)
2. **The population is a time-averaged stock, not a census count.** Chicago's
   $n_j$ is the ACS household count of a 5-year vintage. Ecuador's $n_j$ is
   the mean of 12 monthly firm stocks within the year (a flow-smoothed stock).
   Fitness $w_j = n_j'/n_j$ is the same functional form; only its measurement
   differs. Chicago also carries a second, "real population" column
   (B01003) alongside households; Ecuador sets it equal to the firm count.
3. **Deflation shifts transmission, never selection.** Ecuador's sales are
   deflated to constant-2025 USD by one factor per *year*, common to all
   parishes. In logs that adds the same constant $c_t$ to every $z_j$ in year
   $t$. Every selection term is a **within-year cross-sectional covariance**
   and a covariance is invariant to adding a constant, so all
   $S$ terms are numerically identical with or without deflation; only the
   within-unit growth ($g_j$, hence transmission and the total) loses the
   inflation component. Chicago used nominal ACS dollars (2014 vs 2019
   vintages), so its transmission/total include ~8% cumulative CPI inflation;
   Ecuador's are real.
4. **Annualization horizon.** Chicago: fixed $N=5$ (ACS 2014→2019). Ecuador:
   $N$ = the window length (1, 2, 3 or 6 years), with the same $/N$
   convention; short windows mechanically have noisier $w_j$ and $g_j$.
5. **Survivor set and hierarchy consistency (the one deliberate tightening).**
   Chicago intersects initial/final units **independently at each level** and
   must patch geometry changes (the census tract-split remap step, which never
   fires for Ecuador because DPA codes are stable and nesting is a pure code
   prefix: canton = first 4 digits, province = first 2). Ecuador instead fixes
   a single survivor set of parishes (present with $n>0$ and real sales
   $>\$1{,}000$ in *both* endpoint years) and builds **every** higher level by
   aggregating exactly those parishes. Consequence: the parent traits are, by
   construction, the firm-weighted means of exactly the children entering each
   covariance, so the multilevel telescoping of §2.5 holds *exactly* at every
   scale (sanity-check residuals $\sim 10^{-16}$), rather than up to
   membership mismatches between levels.
6. **The recursive trait correction is constructive rather than corrective.**
   Both pipelines require parent $z$ = population-weighted mean of children's
   $z$ (that is what makes the telescoping exact). Chicago enforces it by
   overwriting census-tabulated parent values
   (`_recursively_calculate_avg_log_income`); Ecuador builds parents that way
   from scratch (`_aggregate_up` in the runner). Same equation, different
   provenance.
7. **Country-normalized NS is derived, not exported, at the canton level.**
   Chicago's exports carry `*_ns_st` at every level because each row lists all
   ancestor IDs. The Ecuador export only carries the immediate parent, so the
   plot scripts recompute $\text{NS}_{ec} = 100\, X / \gamma_{EC}$ from the
   national rollup row (`_ecuador_common.add_country_ns`) — the identical
   formula the engine would have used.

None of these change the decomposition identities; (1)–(4) change what the
symbols *mean* and (5)–(7) make the Ecuadorian identities exact by
construction where Chicago's held only approximately across levels.

---

## 6. Where to verify in the code

| Step                                                             | `runPrice.py`                                                                                       |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| $w_j$, valid-child shares $p_j$, $\bar w_P,\bar z_P,\mu_P$ | `_part_a_initial_child_preparation`, `_part_b_parent_group_aggregates`                            |
| $S^{pop}$ and $S^{inc}$ covariances                          | `_part_d_calculate_selection_terms`                                                                 |
| transmission weights$\phi_j$ (final shares vs initial shares)  | `_part_e_calculate_transmitted_terms`                                                               |
| single- and multi-level identity checks                          | `_sanity_check_handler`, `SanChk_*` columns                                                       |
| recursion up the hierarchy                                       | `calculate_multilevel_price_decomposition`                                                          |
| NS / MSS metrics                                                 | `calculate_ns_and_mss_metrics`                                                                      |
| Ecuador input construction (traits, survivors, aggregation)      | `ecuador_price/decomposition.py`: `_aggregate_up`, `build_levels_for_period`, `build_panel` |
