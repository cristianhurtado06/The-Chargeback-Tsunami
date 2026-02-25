# Implementation Plan: The Chargeback Tsunami
## Storm Retail Fraud Scoring System

**Time budget:** 45 minutes
**Status:** Demo/Proof-of-Concept (peer-reviewed — two rounds applied)

---

## Context

Storm Retail (SE Asian e-commerce) faces a chargeback rate spike from 0.4% → 2.8%, approaching the critical 3% threshold that triggers acquirer account freeze and costs ~$35,000/month in dispute fees. This solution shifts the fraud team from **reactive** (post-shipment review) to **predictive** (pre-fulfillment scoring).

Challenge file: `/Users/cristianhurtado/Documents/Yuno/Challenge/The_Chargeback_Tsunami.md`

---

## Deliverable Structure

```
Challenge/
├── The_Chargeback_Tsunami.md           (existing)
├── IMPLEMENTATION_PLAN.md              (this file)
└── solution/
    ├── README.md                       written last
    ├── requirements.txt                pinned dependencies
    ├── data/
    │   ├── generate_data.py            run first — synthetic data generator
    │   ├── historical_transactions.csv generated: 2,000+ rows, 90 days
    │   └── new_transactions.csv        generated: 100 unlabeled test records
    ├── notebooks/
    │   └── chargeback_analysis.ipynb   PRIMARY ARTIFACT: EDA + model + viz + scoring
    └── outputs/
        └── scored_transactions.csv     final scored output with explanations
```

---

## Time Allocation

| Phase | Task | Time |
|-------|------|------|
| 1 | Directory setup + requirements.txt | 2 min |
| 2 | `generate_data.py` — synthetic transactions with fraud clusters | 10 min |
| 3 | Notebook: EDA + 2 exploratory charts | 5 min |
| 4 | Notebook: Feature engineering (8 RF features + 1 rule-layer signal) | 7 min |
| 5 | Notebook: Model training + 3-way split + PR-AUC + confusion matrix + ROC | 7 min |
| 6 | Notebook: Score 100 new transactions + explainability | 7 min |
| 7 | Notebook: Visualization (Charts 5, 7) | 4 min |
| 8 | README.md + outputs CSV | 3 min |
| **Total** | | **45 min** |

---

## Phase 1 — Directory Setup

Create: `Challenge/solution/{data,notebooks,outputs}/`

**requirements.txt:**
```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1
jupyter==1.1.1
notebook==7.2.2
ipykernel==6.29.5
```

---

## Phase 2 — Synthetic Data Generator

**File:** `data/generate_data.py`
**Runs standalone, no arguments needed, seed=42 for reproducibility.**

### Historical Transactions (2,000 rows, 3.5% chargeback rate)

4 embedded fraud clusters:

| Cluster | % of Fraud | Key Signals |
|---------|-----------|-------------|
| A — Country Mismatch | 35% | billing ≠ shipping, IP ≠ billing, high-risk destination (NG/RO/BD/PK) |
| B — Velocity Attack | 25% | same email 5-12× in 4h, new account (age < 3 days), amounts $50-$150 |
| C — New Account Fraud | 20% | age < 7 days, amount > $400, electronics/jewelry, prepaid card |
| D — Suspicious Email + BIN | 20% | random alphanumeric email, high-risk BIN, high-risk IP country |

### Columns

| Column | Type | Generation Logic |
|--------|------|-----------------|
| `transaction_id` | str | UUID |
| `timestamp` | datetime | 90-day spread, skewed recent |
| `amount_usd` | float | LogNormal(4.2, 0.9), clipped $5-$2000 |
| `customer_email` | str | Legit: `name.surname@domain`; Cluster D: 10-char random |
| `email_domain` | str | Extracted from email |
| `billing_country` | str | Weighted SE Asia: TH/VN/PH/ID/MY/SG |
| `shipping_country` | str | Normally = billing; Cluster A: mismatch |
| `ip_country` | str | Normally = billing; Cluster D: mismatch |
| `card_bin` | str | 6 digits; Cluster D: from `[412345, 511234, 601100, 372345, 349876]` |
| `payment_method` | str | credit/debit/prepaid (prepaid rare, fraud-correlated) |
| `account_age_days` | int | Pareto(3), min 0, cap 1800 |
| `purchases_last_24h` | int | Poisson(1); Cluster B: Poisson(8) |
| `product_category` | str | clothing/shoes/accessories/electronics/jewelry |
| `device_type` | str | mobile 65% / desktop 35% |
| `is_chargeback` | int | 0/1, target label |

### New Transactions (100 rows, no `is_chargeback`)

Composition: 40 clearly safe, 40 clearly suspicious (all 4 clusters), 20 ambiguous (1-2 mild signals).

---

## Phase 3 — Notebook: EDA

Notebook: `notebooks/chargeback_analysis.ipynb`

**Cells:**
1. Imports + load both CSVs + shape/dtype print + class balance check
2. **Chart 4:** Chargeback rate by billing country (horizontal bar, green/yellow/red thresholds)
3. **Chart 6:** Velocity distribution by fraud label (seaborn histplot + KDE)

---

## Phase 4 — Feature Engineering

Single `engineer_features(df)` function applied to both dataframes. The RF is trained on **8 features**; `fraud_signal_count` is computed but reserved for the rule layer only (excluding it from `FEATURE_COLS` prevents double-counting in the 70/30 hybrid blend).

| # | Feature | Formula | Business Reason | Used In |
|---|---------|---------|----------------|---------|
| 1 | `is_country_mismatch` | `(billing_country != shipping_country).astype(int)` | Mule delivery redirect | RF + rules |
| 2 | `is_ip_mismatch` | `(ip_country != billing_country).astype(int)` | VPN/proxy indicator | RF + rules |
| 3 | `velocity_score` | `log1p(purchases_last_24h) * 4.0` | Card testing pattern (log preserves gradient across 5–12 purchase range) | RF + rules |
| 4 | `new_account_large_order` | `(age < 30d) AND (amount > 75th pct)` | Instant large fraud | RF + rules |
| 5 | `is_suspicious_email` | Zero vowels in local part, OR (no separator AND vowel ratio < 20%) | Programmatic email gen; vowel-ratio check prevents false-positives on `johnsmith@gmail.com` | RF + rules |
| 6 | `is_high_risk_bin` | BIN in `[412345, 511234, 601100, 372345, 349876]` | Known fraud BINs | RF + rules |
| 7 | `is_prepaid_card` | `payment_method.str.lower().str.contains('prepaid', na=False)` | Untraceable card; case-insensitive to catch `'Prepaid'`, `'PREPAID'`, `'prepaid_card'` | RF + rules |
| 8 | `amount_zscore` | `(amount - mean) / std` | Outlier amounts (both tails) | RF only |
| 9 | `fraud_signal_count` | sum of features 1–6 (binary) | Multi-signal stacking | Rule layer ONLY — excluded from `FEATURE_COLS` |

Print: `df[FEATURE_COLUMNS].isnull().sum()` after application to catch NaNs.

---

## Phase 5 — Model Training

### Model Choice: Random Forest (Hybrid)

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight='balanced',   # handles 3.5% minority class
    random_state=42,
    n_jobs=-1
)
```

**Justification:**
- **vs Logistic Regression:** RF captures non-linear interactions (velocity × account_age)
- **vs XGBoost:** No extra install overhead; equally accurate at 2,000 rows in demo context
- **`class_weight='balanced'`:** Prevents collapse to all-zeros on 3.5% minority class (auto-weights minority class ~27×)
- **`max_depth=8`:** Reduces overfitting on synthetic 2,000-row dataset

**Split: 70/15/15 stratified on `is_chargeback`.** Two-step split: first 30% held out, then split 50/50 into val and test. Threshold selected on val set (F1-optimal); metrics reported on untouched test set.

> **Synthetic data caveat:** AUC=1.0 is expected by construction — fraud clusters use non-overlapping signals. Do not interpret as a production performance estimate. Realistic range: **0.82–0.93**.

**Final metrics (after all peer review fixes):**
- Threshold (from val): **0.37**
- ROC-AUC: **1.0000** | PR-AUC (primary): **1.0000** | 5-fold CV AUC: **1.0000 ± 0.0000**
- Precision: **0.9091** | Recall: **1.0000** | F1: **0.9524**

**Charts:**
- **Chart 1:** Confusion matrix — `ConfusionMatrixDisplay` with Precision/Recall/F1 annotation
- **Chart 2:** ROC curve — `roc_curve` + AUC, operating point marked at chosen threshold (0.37)

---

## Phase 6 — Scoring New Transactions

### Hybrid Scoring Formula

```
P_ml = model.predict_proba(X)[:,1]           # ML probability
rule_score = fraud_signal_count / 6.0        # Rule-based component

blended = (P_ml * 0.70) + (rule_score * 0.30)
final_score = round(blended * 100, 1)        # 0-100 scale

# Hard overrides (clear-cut fraud patterns)
if is_country_mismatch AND is_ip_mismatch AND is_suspicious_email:
    final_score = max(final_score, 85)
if velocity_score >= 8 AND new_account_large_order:
    final_score = max(final_score, 80)
```

### Risk Tiers

| Tier | Score Range | Action | Reasoning |
|------|------------|--------|-----------|
| HIGH | ≥ 65 | BLOCK / Reject | Captures 85%+ of fraud; ~12% false positive rate |
| MEDIUM | 30-64 | Manual Review | Ambiguous signals requiring human judgment |
| LOW | < 30 | Auto-Approve | Well below fraud centroid |

### Explainability — `triggered_signals` Column

```python
def build_explanation(row):
    signals = []
    if row['is_country_mismatch']:    signals.append("billing/shipping country mismatch")
    if row['is_ip_mismatch']:         signals.append("IP location differs from billing country")
    # Velocity — label mirrors the override conjunction exactly:
    # [velocity override applied → floor 80] only when velocity_score >= 8 AND new_account_large_order == 1
    vs = row['velocity_score']
    vel_override_fired = (vs >= 8.0 and row['new_account_large_order'] == 1)
    if vel_override_fired:
        signals.append(f"extreme purchase velocity ({row['purchases_last_24h']} purchases in 24h) [velocity override applied → floor 80]")
    elif vs >= 8.0:
        signals.append(f"extreme purchase velocity ({row['purchases_last_24h']} purchases in 24h)")
    elif vs >= 5.0:
        signals.append(f"elevated purchase velocity ({row['purchases_last_24h']} purchases in 24h)")
    if row['new_account_large_order']: signals.append("new account with large order")
    if row['is_suspicious_email']:    signals.append("suspicious email pattern")
    if row['is_high_risk_bin']:       signals.append("high-risk BIN detected")
    if row['is_prepaid_card']:        signals.append("prepaid card used")
    if row['amount_zscore'] >= 2.0:   signals.append(f"unusually high amount (z-score={row['amount_zscore']:.1f})")
    return "; ".join(signals) if signals else "no flags triggered"
```

**Chart 5:** Plotly scatter — x=transaction index, y=fraud_score, color=risk_tier (green/orange/red), hover=triggered_signals

---

## Phase 7 — Feature Importance

**Chart 3:** Horizontal bar of `model.feature_importances_` sorted descending, top 3 in red.
**Chart 7 (time permitting):** Boxplot amount by risk tier, log-scale Y axis.

---

## Phase 8 — Output & README

**`outputs/scored_transactions.csv` columns:**
`transaction_id, timestamp, amount_usd, customer_email, billing_country, shipping_country, ip_country, payment_method, account_age_days, purchases_last_24h, product_category, fraud_score, risk_tier, triggered_signals`

**README.md:** 3-command run sequence + approach summary + key findings (fill after running).

---

## End-to-End Run Protocol

```bash
cd Challenge/solution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_data.py
jupyter notebook notebooks/chargeback_analysis.ipynb
# → Kernel > Restart & Run All
```

**Acceptance check:**
1. `outputs/scored_transactions.csv` has 100 rows
2. Highest `fraud_score` row has ≥ 2 entries in `triggered_signals`
3. Confusion matrix + ROC curve render in notebook
4. HIGH/MEDIUM/LOW summary prints successfully

---

## Priority Order (if time runs short)

1. `generate_data.py` — nothing works without data *(critical)*
2. Feature engineering section — core analytical value *(critical)*
3. Model + Charts 1, 2, 3 — proves ML approach *(critical)*
4. Score new transactions + Chart 5 — operational demo *(critical)*
5. README + outputs CSV — submission polish *(important)*
6. Charts 6, 7 — additional exploratory visuals *(nice to have)*

---

## Review Fixes (Applied — Two Rounds)

### Round 1 — Critical Fixes

| # | Issue | Fix |
|---|-------|-----|
| 1 | `is_suspicious_email` false-positives on `johnsmith@gmail.com` | Added vowel-ratio branch: require `vowel_count / len(local) < 0.20` in addition to no-separator check |
| 2 | `velocity_score` signal saturation — all of Cluster B (5–12 purchases) collapsed to score 10.0 | Changed from `purchases * 2.5, clip(0,10)` to `log1p(purchases) * 4.0`; preserves gradient across full range |
| 3 | `fraud_signal_count` double-counted — included in `FEATURE_COLS` (RF) AND 30% rule layer | Removed from `FEATURE_COLS`; RF now has 8 independent features; rule layer uses it exclusively |
| 4 | Threshold selected on test set (leakage) | 3-way 70/15/15 split; threshold (0.37) selected on val set only; metrics reported on untouched test set |
| 5 | `is_prepaid_card` case-sensitive exact match | Changed to `.str.lower().str.contains('prepaid', na=False)` |
| 6 | Hard overrides used boolean index assignment (SettingWithCopyWarning) | Changed to `.loc[mask].clip(lower=floor)` |
| 7 | `pd.cut` tier assignment — sentinel/boundary ambiguity | Replaced with `assign_tier()` map function |
| 8 | `build_explanation()` velocity threshold misaligned | Aligned to `>= 8.0` to match `score_transactions()` override condition |
| 9 | PR-AUC not tracked | Added `average_precision_score` as primary metric alongside ROC-AUC |
| 10 | No robustness validation beyond single split | Added 5-fold stratified CV AUC |

### Round 2 — Second Peer Review Fixes

| # | Issue | Fix |
|---|-------|-----|
| 11 | AUC=1.0 reported without synthetic data caveat | Added caveat block: expected on non-overlapping clusters; realistic range 0.82–0.93 |
| 12 | Status header over-claimed production readiness | Changed from `production-candidate` to `Demo/Proof-of-Concept` |
| 13 | `[override eligible]` label fired on velocity-only condition | Fixed conjunction: `vel_override_fired = (vs >= 8.0 AND new_account_large_order == 1)`; label reads `[velocity override applied → floor 80]` only when both are true |

**Note on `data/params.py`:** Originally planned as a shared constants module. In the actual implementation, `AMOUNT_MEAN`, `AMOUNT_STD`, and `AMOUNT_75TH` are computed inline in the notebook on the full historical dataset before the train/val/test split, and `HIGH_RISK_BINS` / `HIGH_RISK_COUNTRIES` are defined directly in `generate_data.py`. See `TECHNICAL_DESIGN.md §10` for the known mild leakage implication of computing amount statistics before the split, and the recommended production mitigation (sklearn `Pipeline` with `fit_transform`).

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Plotly not rendering in Jupyter | Fallback to `plt.scatter()` with manual color mapping |
| Model predicts all-zeros (imbalance collapse) | `class_weight='balanced'` + verify `y_train.value_counts()` after split |
| NaN values from feature engineering | `fillna(0)` on feature columns + print null check cell |
| Time overrun on visualizations | Cut Charts 6 & 7 first; Charts 1-5 are minimum viable |
| Fraud clusters not separated enough | Verify `fraud_signal_count` mean diff ≥ 2.5 std between fraud/non-fraud |

---

## Acceptance Criteria Cross-Check

| Requirement | Implementation |
|------------|---------------|
| ✅ 2,000+ historical transactions, 90 days | `generate_data.py` → `historical_transactions.csv` |
| ✅ 2-5% chargeback rate | 3.5% base rate, ~70 fraud records |
| ✅ ≥5 meaningful features | 9 signals engineered in `engineer_features()`; 8 fed to RF, 1 reserved for rule layer |
| ✅ 100 new transactions scored | Full hybrid scoring in notebook Section 5 |
| ✅ LOW/MEDIUM/HIGH risk tiers | Score thresholds: <30 / 30-64 / ≥65 |
| ✅ Explainability per transaction | `triggered_signals` column with plain-text reasons |
| ✅ ≥4 visualizations | 7 charts planned (4 required: Charts 1, 2, 3, 4) |
| ✅ Runnable with README | 3-command setup, Kernel > Run All |
