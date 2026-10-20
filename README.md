# German Reunification: A Synthetic Control Analysis

A from-scratch Python implementation of the **synthetic control method** (Abadie, Diamond and Hainmueller, 2015) applied to the 1990 reunification of Germany. West Germany is the treated unit; a weighted combination of 16 OECD countries forms its counterfactual over 1960–2003.

## Implementation

- **Donor weights** estimated by constrained optimisation (`scipy.optimize`): non-negative weights summing to one, chosen to minimise pre-1990 RMSPE on the outcome and on predictors (inflation, trade openness, education, industry share)
- **Missing data** handled with K-nearest-neighbour imputation
- **Outputs**: donor weights, covariate balance, actual vs synthetic path, pointwise and cumulative gaps

## Inference and robustness

| Check | Purpose |
|---|---|
| In-time placebo (fake treatment in 1982) | Tests whether the method finds a "gap" where none should exist |
| In-space placebos (each donor treated in 1990) | Builds the reference distribution of effects |
| Post/pre-RMSPE ratio | Ranks the treated unit against the placebo distribution |

West Germany's post/pre-RMSPE ratio is **30.4**, meaning the post-reunification gap is about thirty times larger than the pre-treatment fit error.

## Repository structure

```
analysis.md   # full analysis: code, weights, gap tables and placebo results
```

## Reproducing

The analysis expects a country-year panel `synthetic_control_germany_robust.csv` with columns `entity_id`, `year_recorded`, `indicator_score` (outcome) and the four predictors. It follows the structure of the Abadie et al. (2015) reunification dataset.

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

## Reference

Abadie, A., Diamond, A., and Hainmueller, J. (2015). Comparative Politics and the Synthetic Control Method. *American Journal of Political Science*, 59(2), 495–510.

## Context

Developed during the MSc in Economics at Sharif University of Technology (Applied Econometrics, Spring 2025).

---
**Shayan Pourmasoomi** · MSc Economics, Sharif University of Technology · [LinkedIn](https://www.linkedin.com/in/shayan-pourmasoomi-8b6206389)
