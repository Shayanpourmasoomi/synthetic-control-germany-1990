# Synthetic Control Method (SCM) — Germany Reunification (1990) | Python

This repository contains a Python implementation of the **Synthetic Control Method (SCM)** to evaluate the economic impact of the **1990 German reunification**.  
The analysis follows the structure of an applied metrics problem set and reproduces standard SCM outputs, including **donor weights, covariate balance, gap plots, and placebo tests**.

> Assignment context: *Problem Set 07 — Synthetic Control Method (SCM), Spring 2025*.

---

## Project Overview

We use a panel dataset for **West Germany** (treated unit) and **16 OECD countries** (donor pool) over **1960–2003** to construct a synthetic counterfactual of West Germany’s GDP (or GDP-like outcome variable), treating **1990** as the intervention year.

The code implements:
- Missing-data handling using **KNN imputation**
- Weight estimation via constrained optimization (weights sum to 1 and are non-negative)
- Visualization of:
  - Actual vs. synthetic outcome
  - Pointwise gap (treated − synthetic)
  - Cumulative gap
- Robustness checks:
  - **In-time placebo** (false intervention year = 1982)
  - **In-space placebo** (each donor unit treated as if it were treated in 1990)
  - **Post/Pre RMSPE ratio** and an empirical placebo-based p-value

---

## Data

Expected input file:

- `synthetic_control_germany_robust.csv`

Required columns:
- `entity_id` (country name / unit id)
- `year_recorded` (year)
- `indicator_score` (outcome, e.g., GDP or GDP proxy)
- Covariates used in matching:
  - `metric_inflation`
  - `trade`
  - `metric_education`
  - `metric_industry_share`

---

## Method (High-level)

Let:
- Treated unit: **West Germany**
- Treatment year: **1990**
- Donor pool: all other countries in the dataset

We estimate donor weights **w** by minimizing pre-treatment RMSPE between the treated outcome and a weighted average of donor outcomes, subject to:
- \( \sum_j w_j = 1 \)
- \( w_j \ge 0 \)

Missing values are imputed using `KNNImputer(n_neighbors=5)`.

---

## Installation

### 1) Create and activate an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
