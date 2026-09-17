# Credit Scoring PD Model

A probability-of-default (PD) model built with logistic regression on Weight-of-Evidence (WOE) transformed features, following a standard credit-risk scorecard workflow: binning → WOE encoding → univariate Gini screening → model fit → deployment scoring with an approve/reject cutoff.

## Dataset

`credit_score.csv` — 87,495 monthly customer records, 25 raw columns.

The target is derived from `CreditScore`:

```python
target = 1 if CreditScore == 'Poor' else 0
```

Roughly 29% of records are labelled bad (`target = 1`).

## Pipeline

### 1. Pre-processing

Dropped identifier and free-text columns that carry no predictive value or are not usable without heavy parsing:
`CustomerID`, `ID`, `Name`, `SSN`, `TypeofLoan`.

Missing values were imputed column-wise — mean for numeric features, mode for categorical:

| Column | Missing |
|---|---|
| MonthlyInhandSalary | 13,176 |
| Amountinvestedmonthly | 7,709 |
| Occupation | 6,178 |
| NumofDelayedPayment | 6,095 |
| ChangedCreditLimit | 1,841 |
| NumCreditInquiries | 1,706 |
| MonthlyBalance | 1,058 |

The data also contains obvious junk values (e.g. `Age` of -500 and 8,698, `InterestRate` up to 5,797, `NumBankAccounts` up to 1,798). Quantile binning absorbs these into the outer bins rather than letting them distort coefficients.

### 2. Train/test split

80/20 split with `random_state=42`, done **before** WOE calculation so that bin edges and WOE values are learned on training data only and then applied to the test set.

### 3. WOE transformation

- **Numeric features** — binned into 4 groups using the training quartiles (Q1, Q2, Q3), with `duplicates='drop'` for features where quantiles coincide.
- **Categorical features** — each category treated as its own bin.

For every bin:

```
WOE = ln( (good_in_bin / total_good) / (bad_in_bin / total_bad) )
```

Bin edges (`bin_maps`) and WOE values (`woe_maps`) are stored so the exact same mapping can be reapplied to test and deployment data.

### 4. Diagnostics

- **Kolmogorov–Smirnov test** run per numeric column — no feature is normally distributed, which is expected and not a problem for logistic regression.
- **Spearman intercorrelation** among WOE features at a 0.7 threshold — **no pairs exceeded the threshold**, so no variables were dropped for multicollinearity.

### 5. Baseline model

Logistic regression on all 20 WOE features:

| Dataset | Gini | Precision | Recall |
|---|---|---|---|
| Train | 61.71 | 0.660 | 0.568 |
| Test | 60.24 | 0.645 | 0.562 |

Test confusion matrix:

|  | Pred. Good | Pred. Bad |
|---|---|---|
| **Actual Good** | 10,866 | 1,565 |
| **Actual Bad** | 2,221 | 2,847 |

Train–test gap of ~1.5 Gini points — no meaningful overfitting.

### 6. Univariate Gini screening

Each WOE variable was fitted alone to measure its standalone discriminatory power.

| Variable | Train Gini | Test Gini |
|---|---|---|
| OutstandingDebt_woe | 0.5160 | 0.5093 |
| InterestRate_woe | 0.5021 | 0.4939 |
| NumCreditInquiries_woe | 0.4473 | 0.4280 |
| Delayfromduedate_woe | 0.4219 | 0.3970 |
| NumCreditCard_woe | 0.3878 | 0.3714 |
| NumofLoan_woe | 0.3552 | 0.3485 |
| NumBankAccounts_woe | 0.3196 | 0.3165 |
| PaymentofMinAmount_woe | 0.3047 | 0.2920 |
| NumofDelayedPayment_woe | 0.2741 | 0.2702 |
| AnnualIncome_woe | 0.2454 | 0.2476 |
| MonthlyBalance_woe | 0.2348 | 0.2454 |
| MonthlyInhandSalary_woe | 0.2144 | 0.2213 |
| PaymentBehaviour_woe | 0.1294 | 0.1279 |
| Age_woe | 0.1275 | 0.1212 |
| TotalEMIpermonth_woe | 0.1063 | 0.1137 |
| Amountinvestedmonthly_woe | 0.1098 | 0.1018 |
| ChangedCreditLimit_woe | 0.0791 | 0.0651 |
| CreditUtilizationRatio_woe | 0.0327 | 0.0485 |
| Occupation_woe | 0.0360 | 0.0293 |
| Month_woe | 0.0121 | 0.0007 |

Selection rules applied:

- Train Gini > 10%
- Test Gini > 10%
- |Train Gini − Test Gini| ≤ 5%

**16 of 20 variables survived.** Dropped: `ChangedCreditLimit`, `CreditUtilizationRatio`, `Occupation`, `Month`.

### 7. Final model

Logistic regression on the 16 retained WOE features:

| Dataset | Gini | Precision | Recall |
|---|---|---|---|
| Train | 59.53 | 0.636 | 0.567 |
| Test | 57.77 | 0.624 | 0.559 |

Test confusion matrix:

|  | Pred. Good | Pred. Bad |
|---|---|---|
| **Actual Good** | 10,722 | 1,709 |
| **Actual Bad** | 2,237 | 2,831 |

### 8. Deployment

Two deployment paths are implemented:

1. **Pre-transformed input** (`deployment_data_with_woe.xlsx`) — WOE columns are already present, so the model scores them directly.
2. **Raw input** (`deployment_data_with_real.xlsx`) — raw values are passed through the stored `bin_maps` and `woe_maps` before scoring. This is the realistic production path.

A cutoff is applied to the predicted PD:

```python
Decision = 'Reject' if PD > 0.25 else 'Approve'
```

## Findings

- **Debt burden and pricing dominate.** `OutstandingDebt` and `InterestRate` alone reach ~0.50 Gini each — nearly as strong as the full model. Interest rate is effectively a proxy for the risk tier the customer was already priced into.
- **Behavioural credit-file variables beat demographics.** Inquiries, delay from due date, number of cards, and number of loans all sit in the 0.35–0.43 range, while `Age` and `Occupation` are near-useless (0.12 and 0.03).
- **`Month` carries essentially zero signal** (test Gini 0.0007), confirming there's no seasonal structure in this panel — each customer simply appears 8 times.
- **Income matters less than expected.** `AnnualIncome` and `MonthlyInhandSalary` land around 0.22–0.25, well below debt and repayment behaviour.
- **Dropping the 4 weak variables cost ~2.5 Gini points** (60.24 → 57.77 on test). The trade is a simpler, more stable, more explainable scorecard — a reasonable exchange in a credit-risk setting, though if raw performance is the goal the full model is better.
- **Recall is the weak spot.** At the default 0.5 threshold the model catches only ~56% of bad customers. In practice the threshold should be tuned to the business cost of a false approval versus a lost good customer, which is exactly why the deployment step uses a 0.25 cutoff rather than 0.5.
- **No multicollinearity** among WOE features at |ρ| ≥ 0.7, so coefficient signs remain interpretable.

## Repo structure

```
.
├── Logistic_regression_praktiki_tapshiriq.ipynb   # full analysis
├── credit_score.csv                               # training data
├── deployment_data_with_woe.xlsx                  # deployment set (pre-transformed)
└── deployment_data_with_real.xlsx                 # deployment set (raw values)
```

## Requirements

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
openpyxl
```

```bash
pip install -r requirements.txt
jupyter notebook
```

> **Note:** file paths in the notebook are currently absolute local paths. Change them to relative paths before running elsewhere.

## Possible improvements

- Clean the junk values (negative ages, 5,797% interest rates) explicitly instead of relying on binning to absorb them.
- Enforce monotonic WOE bins for numeric features rather than fixed quartiles.
- Split by customer rather than by row — the same customer appears in both train and test, which leaks information and likely inflates the reported Gini.
- Report Information Value alongside univariate Gini for variable selection.
- Convert PD into a scaled scorecard (points, PDO) for business use.
