# Outlier / leverage check — Ecuador Price decomposition (_pop path)

Question: is the **negative selection** driven by a few outliers, even though the trait is `log(real sales)`?

## 1. Logging fixes the trait, not the fitness

Selection is `cov(relative fitness, trait)`. The trait is logged; the fitness (firm-count growth ratio) is not.

| variable (full window) | skew | kurtosis | median | p99 | max |
|---|---:|---:|---:|---:|---:|
| trait = log(real sales) | 0.14 | -0.44 | 13.67 | 21.46 | 24.52 |
| fitness = firms_f/firms_i | 5.04 | 44.92 | 1.71 | 6.50 | 17.58 |

The trait is near-symmetric (skew 0.14); the fitness is heavy right-tailed (skew 5.04, max 17.6x). So any outlier risk in selection lives on the **fitness** side, which `log` does nothing about.

## 2. The negative selection is broad, not one unit

Pooled single-scale selection = -0.02702 (it equals the multiscale Cumulative_Sel = -0.02702 exactly -- law of total covariance). Share of total |contribution| from the top parishes:

| top-k parishes | share of |selection| |
|---|---:|
| 1 | 19.2% |
| 5 | 27.3% |
| 10 | 32.1% |
| 20 | 39.5% |
| 8 | 30.3% |

The top contributors are the country's economic centres, not data outliers -- their fitness is moderate (well inside the p1/p99 clip band), they matter because of their firm share:

| parish | province | firm share | fitness | trait | contribution |
|---|---|---:|---:|---:|---:|
| 090150 | 09 (Guayas) | 0.303 | 1.57 | 24.45 | -0.01144 |
| 170150 | 17 (Pichincha) | 0.262 | 1.70 | 24.52 | -0.00205 |
| 090650 | 09 (Guayas) | 0.003 | 3.97 | 20.02 | -0.00137 |

Guayaquil (90150) and Quito (170150) alone hold ~56% of all formal firms; their below-average firm-count growth on top of the highest sales is what makes selection negative. Small parishes with explosive fitness (e.g. 8->101 firms) exist but carry tiny firm shares, so the firm-share weighting caps their leverage -- which the re-run confirms.

## 3. The dominant piece is between provinces

Most-negative provinces in the between-province selection term (rows sum to -0.02271):

| province | firm share | fitness | trait | contribution |
|---|---:|---:|---:|---:|
| Guayas | 0.344 | 1.65 | 23.93 | -0.00543 |
| Morona Santiago | 0.004 | 2.34 | 15.28 | -0.00169 |
| Orellana | 0.006 | 2.36 | 17.64 | -0.00152 |
| Zamora Chinchipe | 0.005 | 2.15 | 15.04 | -0.00149 |
| Pichincha | 0.341 | 1.71 | 23.51 | -0.00145 |
| Los Rios | 0.013 | 2.03 | 18.34 | -0.00144 |

Large, high-trait provinces with below-average fitness pull selection down — this is the convergence/reallocation story, not an outlier.

## 4. Robustness: re-run the engine on cleaned fitness (full window)

| variant | selection_total | between-prov | transmission |
|---|---:|---:|---:|
| raw | -0.02702 | -0.02271 | 0.04963 |
| winsorized p1/p99 | -0.02646 | -0.02236 | 0.04944 |
| trimmed p1/p99 | -0.02624 | -0.02213 | 0.04918 |

**Conclusion:** if the sign and the between-province dominance hold under winsorized and trimmed fitness, the negative selection is a structural feature of Ecuador's formal firms, not an artifact of a few high-growth small parishes. The untrimmed numbers remain the reported result.

## 5. Robustness: the SALES_FLOOR (near-empty parishes)

Current `SALES_FLOOR = 1000`. The lowest floors admit parishes with absurdly small annual real sales (min ~$5/yr at a $1 floor). Re-running the engine across floors (full window):

| sales floor ($/yr) | survivors | selection_total | between-prov | transmission |
|---:|---:|---:|---:|---:|
| 1 | 859 | -0.02629 | -0.02282 | 0.05006 |
| 100 | 829 | -0.02722 | -0.02283 | 0.04976 |
| 1,000 | 804 | -0.02702 | -0.02271 | 0.04963 |
| 10,000 | 712 | -0.02691 | -0.02245 | 0.04910 |
| 100,000 | 576 | -0.02760 | -0.02227 | 0.04847 |

Raising the floor from $1 to $100k drops 283 of 859 parishes and moves selection from -0.02629 to -0.02760 -- it stays negative and, as expected, the tiny-sales parishes were contributing *positively*, so removing them makes selection slightly **more** negative. The finding is not propped up by near-empty parishes.