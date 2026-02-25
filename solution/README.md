# Chargeback Tsunami — Fraud Scoring System

## Problem Statement

Storm Retail's chargeback rate spiked from **0.4% → 2.8%** (acquirer freeze threshold: 3%).
This solution shifts from reactive dispute handling to **predictive fraud prevention** — scoring every transaction before fulfillment.

---

## Quick Start (3 commands)

```bash
# 1. Set up environment
cd Challenge/solution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic data
python data/generate_data.py

# 3. Open and run the notebook
jupyter notebook notebooks/chargeback_analysis.ipynb
# → Kernel > Restart & Run All
```

**Output:** `outputs/scored_transactions.csv` — 100 scored new transactions ready for review.

---

## Approach Summary

### Data
- **2,000 historical transactions** with 3.5% chargeback rate and 4 embedded fraud clusters
- **100 unlabeled new transactions** (40 safe / 40 suspicious / 20 ambiguous)

### Fraud Clusters Modeled
| Cluster | Pattern | Share of Fraud |
|---------|---------|---------------|
| A | Billing/shipping country mismatch + high-risk destination | 35% |
| B | Velocity attack (5–12 purchases in 4h, new account <3 days) | 25% |
| C | New account (<7 days) + high amount (>$400) + electronics | 20% |
| D | Suspicious email + high-risk BIN + high-risk IP country | 20% |

### Feature Engineering (8 RF features + 1 rule-layer signal)
| Feature | Formula | Used In |
|---------|---------|---------|
| `is_country_mismatch` | billing ≠ shipping country | RF + rules |
| `is_ip_mismatch` | ip_country ≠ billing country | RF + rules |
| `velocity_score` | `log1p(purchases_last_24h) * 4.0` | RF + rules |
| `new_account_large_order` | age < 30d AND amount > 75th pct | RF + rules |
| `is_suspicious_email` | zero vowels in local part, OR (no separator AND vowel ratio < 20%) | RF + rules |
| `is_high_risk_bin` | BIN in known fraud list | RF + rules |
| `is_prepaid_card` | `payment_method.str.lower().str.contains('prepaid')` | RF + rules |
| `amount_zscore` | (amount − mean) / std | RF only |
| `fraud_signal_count` | sum of 6 binary features above | Rule layer only — excluded from RF |

### Model
- **Random Forest** (200 trees, max_depth=8, class_weight='balanced') — trained on **8 features** (`fraud_signal_count` excluded from RF; reserved for rule layer)
- **3-way stratified split:** 70% train / 15% val / 15% test; threshold selected on val set only
- **Threshold:** 0.37 (F1-optimal on validation set; test set never touched during selection)
- **Hybrid scoring:** 70% ML probability + 30% rule-based signal score
- **Hard overrides:** triple-mismatch patterns → score ≥ 85; velocity (≥ 8 purchases) + new account → score ≥ 80

### Risk Tiers & Actions
| Tier | Score | Action |
|------|-------|--------|
| HIGH | ≥ 65 | BLOCK — do not fulfill |
| MEDIUM | 30–64 | MANUAL REVIEW — ops team flags |
| LOW | < 30 | AUTO-APPROVE |

---

## Key Findings

> **Demo/Proof-of-Concept** — metrics below are from a synthetic dataset with non-overlapping fraud clusters. AUC=1.0 is expected by construction, not a production performance estimate. Realistic range on real data: **0.82–0.93**. See `TECHNICAL_DESIGN.md §6` for full caveat.

| Metric | Value |
|--------|-------|
| Split | 70% train / 15% val / 15% test (stratified) |
| Threshold | 0.37 (selected on val set) |
| ROC-AUC | 1.0000 (synthetic data — see caveat above) |
| PR-AUC (primary metric) | 1.0000 (synthetic data — see caveat above) |
| 5-fold CV AUC | 1.0000 ± 0.0000 |
| Precision | 0.9091 |
| Recall | 1.0000 |
| F1 Score | 0.9524 |
| HIGH risk transactions (new) | 49 / 100 → BLOCK |
| MEDIUM risk transactions (new) | 8 / 100 → MANUAL REVIEW |
| LOW risk transactions (new) | 43 / 100 → AUTO-APPROVE |
| Top fraud signal | See Chart 3 (feature importances) |

---

## File Structure

```
solution/
├── README.md                        ← You are here
├── requirements.txt                 ← Pinned dependencies
├── data/
│   ├── generate_data.py             ← Run first — generates both CSVs
│   ├── historical_transactions.csv  ← 2,000 labeled transactions
│   └── new_transactions.csv         ← 100 unlabeled transactions
├── notebooks/
│   └── chargeback_analysis.ipynb   ← Main artifact (7 charts + model)
└── outputs/
    └── scored_transactions.csv      ← Final output after running notebook
```

---

## Design Decisions

**Why Random Forest over XGBoost?**
No additional install overhead; performance is equivalent at 2,000 rows. The `class_weight='balanced'` parameter handles the severe 3.5% class imbalance without resampling.

**Why 8 features in the RF (not 9)?**
`fraud_signal_count` is a linear combination of the 6 other binary features already fed to the RF. Including it in `FEATURE_COLS` caused double-counting in the 70/30 hybrid blend: the RF already encoded the aggregate, and then the 30% rule component added it again. Removing it from the RF restores the blend to two genuinely independent information sources.

**Why a 3-way split with a dedicated validation set?**
Selecting the classification threshold by iterating over the test set is a form of data leakage — the reported F1 would be the best achievable on that specific random sample, not a conservative generalization estimate. The validation set (15%) is used exclusively for threshold selection. The test set (15%) is never examined until final evaluation.

**Why PR-AUC as the primary metric?**
At 3.5% fraud prevalence, ROC-AUC is dominated by true negatives and can look high even when the model misses significant fraud. Precision-Recall AUC directly measures how well the model balances catching fraud (recall) against avoiding false blocks (precision) — the trade-off that matters operationally.

**Why hybrid scoring (ML + rules)?**
Pure ML can miss novel attack patterns not present in training data. The rule component (30%) ensures that strong signal stacking is always elevated — acting as a safety net with full explainability. The two inputs are now genuinely independent: the RF operates on 8 features; the rule layer operates on `fraud_signal_count` (the raw binary signal sum).

**Why explainability matters here?**
Fraud analysts need to justify blocks to customers and card networks. The `triggered_signals` column provides plain-text reasoning for every scored transaction. The velocity override label (`[velocity override applied → floor 80]`) appears only when both conditions of the override conjunction are satisfied (`velocity_score >= 8 AND new_account_large_order == 1`), ensuring the explanation accurately reflects when the score floor was actually applied.
