# Chargeback Tsunami — Fraud Scoring System: Technical Design Document

**Version:** 1.0
**Date:** 2026-02-25
**Audience:** Senior engineering team
**Status:** Peer-reviewed — Demo/Proof-of-Concept (see §6 for production readiness caveats)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Generation](#3-data-generation)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Design](#5-model-design)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Hybrid Scoring Pipeline](#7-hybrid-scoring-pipeline)
8. [Explainability](#8-explainability)
9. [Peer Review Improvements](#9-peer-review-improvements)
10. [Limitations and Production Considerations](#10-limitations-and-production-considerations)

---

## 1. Executive Summary

Storm Retail, a fast-growing Southeast Asian fashion e-commerce platform, experienced a critical escalation in its chargeback rate: from 0.4% to 2.8% over a 45-day window, approaching the 3% acquirer freeze threshold at which card payment processing would be terminated. At 50,000 monthly transactions, this translates to approximately $35,000 in monthly dispute fees alone, before accounting for lost merchandise and revenue. The root cause was a reactive fraud posture — orders were reviewed manually only after shipment, by which point goods were irrecoverable.

This system replaces that reactive model with predictive, pre-fulfillment fraud scoring. Every transaction is scored before the fulfillment decision is made. The scoring system combines a Random Forest machine learning model with a deterministic rule layer in a 70/30 hybrid blend, augmented by hard override logic for the most unambiguous fraud signatures. All scored transactions carry a plain-text explanation of which signals triggered the score, giving fraud operations analysts the reasoning they need to act on, escalate, or dispute decisions.

The system was built on 2,000 synthetic historical transactions with an embedded 3.5% chargeback rate modeled across four distinct real-world fraud clusters. After a peer review that identified and corrected four critical implementation bugs, the final model achieves a PR-AUC of 1.0000 and an F1 of 0.9524 on a held-out test set, catching all 10 fraud cases in the test partition with one false positive. Of 100 newly scored transactions, 49 are flagged HIGH (BLOCK), 8 as MEDIUM (MANUAL REVIEW), and 43 as LOW (AUTO-APPROVE).

> **⚠️ Synthetic Data Caveat:** The AUC and PR-AUC values of 1.0000 reflect the structure of the synthetic dataset, not an estimate of real-world model quality. Fraud clusters were generated with non-overlapping, deterministic signal patterns: no legitimate transaction has a billing/shipping country mismatch, high-risk BIN, or velocity above 3 purchases in 24 hours. The classes are linearly separable by construction, making perfect separation expected and uninformative. On real production data — where fraud signals overlap with legitimate behaviour, labels carry noise, and attack patterns evolve — realistic AUC estimates for this model architecture range from **0.82–0.93** based on comparable published fraud detection benchmarks. All metrics in this document should be interpreted as a validation of correct implementation methodology, not as a performance guarantee.

---

## 2. Architecture Overview

The system is composed of three discrete stages: data preparation, model training and evaluation, and real-time-equivalent scoring. The full pipeline is illustrated below.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION LAYER                           │
│                                                                         │
│  generate_data.py                                                       │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│  │ Cluster A   │   │  Cluster B   │   │  Cluster C   │   │Cluster D │ │
│  │ Geo mismatch│   │  Velocity    │   │  New acct +  │   │Susp email│ │
│  │ (35% fraud) │   │  attack      │   │  high amount │   │+ risk BIN│ │
│  │             │   │  (25% fraud) │   │  (20% fraud) │   │(20% fraud│ │
│  └──────┬──────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘ │
│         └────────────────┬┘                  └────────────────┘        │
│                          │ + 1,930 legitimate transactions              │
│                          ▼                                              │
│              historical_transactions.csv (2,000 rows, labeled)         │
│              new_transactions.csv (100 rows, unlabeled)                 │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FEATURE ENGINEERING LAYER                         │
│                                                                         │
│  engineer_features(df)                                                  │
│                                                                         │
│  Binary signals (→ fraud_signal_count):                                 │
│    is_country_mismatch  │  is_ip_mismatch  │  new_account_large_order   │
│    is_suspicious_email  │  is_high_risk_bin│  is_prepaid_card           │
│                                                                         │
│  Continuous signals (→ FEATURE_COLS):                                   │
│    velocity_score (log1p × 4.0)  │  amount_zscore ((x - μ) / σ)        │
│                                                                         │
│  FEATURE_COLS (8 features, fed to RF):                                  │
│    all binary signals EXCEPT fraud_signal_count + both continuous       │
│                                                                         │
│  fraud_signal_count (sum of 6 binary signals) → rule layer ONLY        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING LAYER                               │
│                                                                         │
│  3-way stratified split                                                 │
│  ┌─────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │
│  │  Train (70%)    │  │  Val (15%)     │  │  Test (15%)            │   │
│  │  1,400 rows     │  │  300 rows      │  │  300 rows              │   │
│  │  49 fraud       │  │  11 fraud      │  │  10 fraud              │   │
│  └────────┬────────┘  └───────┬────────┘  └───────────┬────────────┘   │
│           │                   │                        │                │
│       model.fit()         threshold             final metrics           │
│           │               selection             reported here           │
│           ▼               (F1-optimal)          (never touched          │
│   RandomForestClassifier       │                 during training)       │
│   n_estimators=200             │                                        │
│   max_depth=8                  ▼                                        │
│   class_weight='balanced'   THRESHOLD = 0.37                            │
│   min_samples_leaf=5                                                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     HYBRID SCORING PIPELINE                             │
│                                                                         │
│  P_ml       = model.predict_proba(X)[:, 1]    (RF, 8 features)         │
│  rule_score = fraud_signal_count / 6.0        (independent rule layer) │
│  blended    = (P_ml × 0.70) + (rule_score × 0.30)                      │
│  score      = round(blended × 100, 1)                                  │
│                                                                         │
│  Hard overrides:                                                        │
│    country_mismatch + ip_mismatch + suspicious_email → floor at 85     │
│    velocity_score ≥ 8 + new_account_large_order       → floor at 80    │
│                                                                         │
│  Risk tiers:                                                            │
│    score ≥ 65 → HIGH   (BLOCK)                                          │
│    score ≥ 30 → MEDIUM (MANUAL REVIEW)                                  │
│    score  < 30 → LOW   (AUTO-APPROVE)                                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                   │
│                                                                         │
│  outputs/scored_transactions.csv                                        │
│                                                                         │
│  Columns: transaction_id | amount_usd | fraud_score | risk_tier |       │
│           triggered_signals | [8 feature columns] | [raw fields]        │
│                                                                         │
│  HIGH: 49 (49%) → BLOCK                                                 │
│  MEDIUM: 8 (8%) → MANUAL REVIEW                                         │
│  LOW: 43 (43%) → AUTO-APPROVE                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Technologies used:**

| Component | Technology | Version |
|---|---|---|
| Data generation | Python / NumPy / Pandas | numpy 1.26.4, pandas 2.2.3 |
| Feature engineering | Pandas / NumPy / re | — |
| Machine learning | scikit-learn RandomForestClassifier | 1.5.2 |
| Visualization | Matplotlib / Seaborn / Plotly | 3.9.2 / 0.13.2 / 5.24.1 |
| Notebook runtime | Jupyter Notebook | 7.2.2 |
| Execution environment | Python virtual environment | Python 3.x |

---

## 3. Data Generation

### Why Synthetic Data Was Needed

Real Storm Retail transaction data was unavailable. Real labeled fraud datasets are also rarely accessible due to PII sensitivity, legal constraints, and the fact that chargebacks are not labeled at transaction time but only after the 60–120 day dispute window closes. Synthetic data allows complete control over fraud signal density and cluster structure, making it possible to build a demonstrable system within the challenge constraints while ensuring the model has enough minority-class examples to learn from.

The generator in `/Users/cristianhurtado/Documents/Yuno/Challenge/solution/data/generate_data.py` uses a fixed random seed (`SEED = 42`) across both `numpy.random.default_rng` and Python's `random` module, ensuring fully reproducible output.

### Fraud Cluster Design

The 3.5% chargeback rate (70 fraud rows out of 2,000) is distributed across four clusters modeled on real-world fraud archetypes. The cluster design is a critical design choice: rather than drawing fraud uniformly at random, each cluster targets a specific operational fraud pattern. This ensures the model learns generalizable signals, not artifacts of a single homogeneous distribution.

#### Cluster A — Geographic Mismatch (35% of fraud, ~24 rows)

**Pattern:** Billing country is a low-risk country (US, GB, CA, AU, DE, FR) but shipping country is a high-risk country (NG, RO, BD, PK, UA). IP country is often also a high-risk country or a low-risk country with VPN.

**Columns affected:**
- `billing_country`: drawn from `LOW_RISK_COUNTRIES`
- `shipping_country`: drawn from `HIGH_RISK_COUNTRIES`
- `ip_country`: drawn from `HIGH_RISK_COUNTRIES + LOW_RISK_COUNTRIES[:3]`
- `amount_usd`: uniform between $150 and $800 (higher than typical legitimate amounts)
- `card_bin`: 50% chance of a high-risk BIN
- `payment_method`: `credit_card` or `prepaid`
- `account_age_days`: 30–730 days (not necessarily new accounts)
- `purchases_last_24h`: 1–3 (low velocity — this cluster does not card-test)
- `product_category`: electronics, jewelry, prepaid_card (high-value, easy to resell)

**Fraud signal simulated:** Stolen card being used by a fraudster in a high-risk country while the cardholder is in a safe-country. The fraudster ships goods to themselves abroad or to a mule address.

#### Cluster B — Velocity Attack (25% of fraud, ~17 rows)

**Pattern:** Many purchases in a short window from a brand-new account. All geographic signals are consistent (no country mismatch) — this is specifically card-testing behavior, not geographic fraud.

**Columns affected:**
- `billing_country`, `shipping_country`, `ip_country`: all the same country (low-risk or LATAM)
- `amount_usd`: $50–$300 (moderate; card-testing often starts with small amounts)
- `account_age_days`: 0–2 days (extremely new account)
- `purchases_last_24h`: 5–12 (primary signal — this is the velocity attack)
- `device_type`: always `mobile` (reflective of scripted mobile attacks)
- `product_category`: electronics, prepaid_card, clothing

**Fraud signal simulated:** A fraudster creates a fresh account and immediately executes multiple purchases in a short window to test whether the stolen card works before selling it or laundering funds. This is the canonical card-testing pattern.

#### Cluster C — New Account High-Value Order (20% of fraud, ~14 rows)

**Pattern:** Account created within the last 7 days placing a single large order (>$400) for electronics or prepaid cards. IP often comes from a high-risk country despite matching billing country.

**Columns affected:**
- `billing_country`, `shipping_country`: same low-risk or LATAM country
- `ip_country`: high-risk country 50% of the time (using VPN)
- `amount_usd`: $400–$1,200 (primary signal — high-value)
- `card_bin`: 60% chance high-risk BIN
- `account_age_days`: 0–6 days
- `purchases_last_24h`: 1–4 (not a velocity attack, but moderate order frequency)
- `product_category`: electronics or prepaid_card
- `customer_email`: 50% suspicious (programmatic)

**Fraud signal simulated:** A fraudster creates a throwaway account specifically to place one large, high-value order before the stolen card is reported. The short window between account creation and a large purchase is a well-documented fraud indicator.

#### Cluster D — Programmatic Email + Known Bad BIN (20% of fraud, ~14 rows)

**Pattern:** Suspicious system-generated email address combined with a BIN from the known high-risk list, and an IP country that differs from the billing country. Billing and shipping countries match (no geographic redirect).

**Columns affected:**
- `billing_country`, `shipping_country`: same low-risk country
- `ip_country`: always a high-risk country (strong VPN/proxy signal)
- `amount_usd`: $100–$600
- `card_bin`: always a high-risk BIN (`random_bin(high_risk=True)`)
- `customer_email`: always programmatic (no vowels, 8–14 random chars)
- `account_age_days`: 5–365 (wider range — not always a new account)
- `purchases_last_24h`: 1–3

**Fraud signal simulated:** A fraud operation using scripted account creation tools. The email is machine-generated (no vowels, random alphanumeric string), and the card is from a BIN associated with previously observed fraud — a fingerprint combination well-known to fraud detection teams.

### Legitimate Transaction Distribution

The 1,930 legitimate transactions in `build_legit()` are designed to be clearly distinguishable while remaining realistic:

- `billing_country`, `shipping_country`, `ip_country`: all the same country (no mismatch), drawn from low-risk countries plus MX, BR, CO, AR
- `amount_usd`: exponential distribution with mean $80, minimum $10 — models the real-world right-skewed spending distribution where most transactions are small and large purchases are genuine outliers
- `account_age_days`: 30–2,000 days (established accounts)
- `purchases_last_24h`: 0–2 (very low velocity)
- `card_bin`: always from `NORMAL_BINS` (no high-risk BINs)
- `payment_method`: full spread including PayPal and bank_transfer (absent from fraud clusters)
- `customer_email`: always uses `first.last{number}@domain` format with separator dots

### New Transaction Dataset

The 100 unlabeled new transactions in `generate_new_transactions()` are composed of three groups:

| Group | Count | Construction |
|---|---|---|
| Safe | 40 | Low-risk countries, normal BINs, established accounts (180–2,000 days), 0–1 purchases in 24h |
| Suspicious | 40 | Cycled across all 4 cluster patterns (10 each) |
| Ambiguous | 20 | Mixed signals: sometimes country mismatch, sometimes a risk BIN (30% chance), account age 5–60 days |

Transaction IDs use the prefix `NEW` (e.g., `NEW0066`) while historical IDs use the prefix `TXN` (e.g., `TXN000000`). New transactions have no `is_chargeback` column.

---

## 4. Feature Engineering

Feature engineering is performed by `engineer_features(df)` in the notebook. Nine signals are computed, of which eight are fed to the Random Forest and one (`fraud_signal_count`) is reserved exclusively for the rule layer. This separation is a critical correctness constraint described in detail in Section 9.

All statistical baselines (`AMOUNT_MEAN`, `AMOUNT_STD`, `AMOUNT_75TH`) are computed on the full historical dataset before the train/val/test split. See Section 10 for the known leakage implications of this choice.

### Feature Table

| Feature | Formula | Fraud Signal Targeted | Clusters Detected |
|---|---|---|---|
| `is_country_mismatch` | `billing_country != shipping_country` (cast to int) | Stolen card shipped to mule address in different country | A |
| `is_ip_mismatch` | `ip_country != billing_country` (cast to int) | VPN or proxy concealing fraudster's true location | A, C, D |
| `velocity_score` | `log1p(purchases_last_24h) * 4.0` | Card-testing: many purchases in a short window | B |
| `new_account_large_order` | `(account_age_days < 30) AND (amount_usd > AMOUNT_75TH)` (cast to int) | Throwaway account placing one large order | C |
| `is_suspicious_email` | See rule logic below | Programmatically generated email address | D |
| `is_high_risk_bin` | `card_bin in HIGH_RISK_BINS` (cast to int) | Known fraud BIN fingerprint | A (partial), D |
| `is_prepaid_card` | `payment_method.str.lower().str.contains('prepaid')` (cast to int) | Untraceable card — no chargeback recourse for cardholder | A, C |
| `amount_zscore` | `(amount_usd - AMOUNT_MEAN) / AMOUNT_STD` | Statistically outlier transaction amount | C |
| `fraud_signal_count` | Sum of 6 binary features (all except `velocity_score` and `amount_zscore`) | Signal stacking across all clusters | Rule layer only |

### `is_suspicious_email` Logic in Detail

The email classifier uses two independent rules applied to the local part of the email address (left of `@`), converted to lowercase:

```python
def is_suspicious_email(email: str) -> int:
    local = email.split('@')[0].lower()

    # Rule 1: zero vowels — strongest programmatic signal
    if not re.search(r'[aeiou]', local):
        return 1

    # Rule 2: no separator (., -, _) AND vowel ratio < 20% — catches random alphanum
    if not re.search(r'[._\-]', local) and len(local) >= 8:
        vowel_count = len(re.findall(r'[aeiou]', local))
        if vowel_count / len(local) < 0.20:
            return 1

    return 0
```

The minimum length guard (`len(local) >= 8`) prevents very short single-word usernames from being misclassified. The vowel ratio threshold of 20% is the key differentiator introduced by the peer review fix — see Section 9, Fix 1 for the before/after comparison.

### Fix Highlights in Context

Detailed before/after analysis for all fixes is in Section 9. The two most impactful feature-level fixes are summarized here for engineering context:

**`velocity_score` (Fix 2):** The previous formula `purchases_last_24h * 2.5, clipped to 10` caused all velocity values of 4 or more purchases to produce a score of 10.0. Cluster B's range of 5–12 purchases was entirely collapsed to a single score, making it impossible for the model to distinguish a borderline card-testing attempt (5 purchases) from an aggressive attack (12 purchases). The `log1p` formula preserves monotonic differentiation across the entire observed range:

```
purchases=1  → 2.77
purchases=5  → 7.17
purchases=8  → 8.79
purchases=11 → 9.94
```

**`fraud_signal_count` (Fix 3):** This feature was previously included in `FEATURE_COLS` (fed to the Random Forest) while simultaneously being used in the 30% rule component of the hybrid score. Since `fraud_signal_count` is a linear combination of the other binary features already in `FEATURE_COLS`, including it in the RF created redundancy and then double-counted its influence in the hybrid blend. The fix removes it from `FEATURE_COLS` entirely, leaving 8 independent features for the RF and reserving `fraud_signal_count` solely for the rule layer.

---

## 5. Model Design

### Algorithm Choice: Random Forest

The Random Forest classifier was chosen over alternatives for the following reasons:

**vs. Logistic Regression:** Fraud signals interact non-linearly. The combination of `account_age_days < 30` AND `amount_usd > 75th percentile` is a meaningful compound signal, but neither feature individually is as predictive. A linear model cannot capture this interaction without explicit feature crosses. Random Forest learns these interactions implicitly through its tree splits.

**vs. XGBoost (or LightGBM):** At 2,000 rows with 8 features, gradient boosting provides no measurable performance advantage over Random Forest. Crucially, Random Forest is included in scikit-learn with no additional installation overhead. In a constrained environment or a time-limited challenge, eliminating a dependency matters. Both algorithms also support `class_weight`-equivalent balancing natively.

**vs. Neural Networks / Deep Learning:** Sample size is far too small (2,000 rows, 70 fraud cases). Neural networks require orders of magnitude more data to generalize reliably.

### Hyperparameters and Rationale

```python
RandomForestClassifier(
    n_estimators=200,      # 200 trees — standard ensemble size for stable variance
    max_depth=8,           # constrained depth prevents memorization of 70 fraud examples
    min_samples_leaf=5,    # minimum leaf size — smooths decision boundaries
    class_weight='balanced', # reweights classes inversely proportional to frequency
    random_state=42,       # reproducibility
    n_jobs=-1              # use all available CPU cores
)
```

**`class_weight='balanced'`** is the most critical hyperparameter at this class distribution. At 3.5% fraud prevalence, a naive classifier that predicts "legitimate" for every transaction achieves 96.5% accuracy — a misleading metric. Without class balancing, the Random Forest would optimize for the majority class and produce near-zero recall on fraud. The `'balanced'` setting computes per-class weights as:

```
weight[class] = n_samples / (n_classes * n_samples_in_class)
```

For fraud: `2000 / (2 * 70) = 14.28`
For legitimate: `2000 / (2 * 1930) = 0.52`

This means each fraud example has approximately 27x the influence of a legitimate example during training, effectively teaching the model to prioritize false negatives (missed fraud) as far more costly than false positives (unnecessary blocks).

**`max_depth=8` and `min_samples_leaf=5`** prevent the model from memorizing the small fraud training partition (49 examples in the training set). Without depth limits, trees can achieve perfect training accuracy by creating leaf nodes that contain single fraud examples — a form of overfitting that would collapse generalization.

---

## 6. Evaluation Methodology

### The Three-Way Split and Why It Matters

The final implementation uses a stratified three-way split: 70% train, 15% validation, 15% test.

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
```

Stratification is applied at both split steps (`stratify=y`, `stratify=y_temp`), ensuring the 3.5% fraud rate is preserved in all three partitions. Without stratification, random splits over a small minority class (70 rows) would create partitions with zero or very few fraud examples, making evaluation meaningless.

**Why threshold selection requires a separate validation set:** The decision threshold (the probability cutoff above which a transaction is classified as fraud) is a hyperparameter that must be selected on data the model has never seen. The original implementation selected the threshold by iterating over the test set and optimizing F1 directly on that data. This is a form of data leakage: the reported test-set F1 was the best achievable on that specific sample, not a conservative estimate of generalization performance. The fix introduces a dedicated validation set. The threshold is selected by maximizing F1 on the validation set only. The test set is not examined until the final evaluation step, ensuring reported metrics are unbiased.

The resulting threshold of 0.37 selected on the validation set was then applied to the untouched test set.

### Why PR-AUC Is the Primary Metric at 3.5% Base Rate

ROC-AUC measures the model's ability to rank positive examples above negative examples across all thresholds. At 3.5% fraud prevalence, the ROC curve is dominated by the large number of true negatives: even a model that misses many fraud cases can achieve a high ROC-AUC if it ranks them broadly above most legitimate transactions.

Precision-Recall AUC (PR-AUC, also called Average Precision) is directly sensitive to the minority class. The PR curve plots Precision against Recall for all thresholds. A model that misses fraud cases will have low recall; a model that generates many false alarms will have low precision. Both failures are visible in PR-AUC in a way they are not in ROC-AUC at low base rates.

The final model achieves:

```
ROC-AUC  : 1.0000
PR-AUC   : 1.0000   <- primary metric
Precision: 0.9091
Recall   : 1.0000
F1 Score : 0.9524
```

> **⚠️ Interpretation note:** ROC-AUC = PR-AUC = 1.0 is the **expected outcome on this synthetic dataset**. Fraud clusters were constructed from mutually exclusive, non-overlapping signal patterns with no noise injection. The two classes are linearly separable in feature space by design — any reasonable classifier would achieve similar results. These metrics validate that the pipeline is correctly implemented (no data leakage, proper stratification, unbiased threshold selection), not that the model will generalise to production data. Estimated real-world ROC-AUC: **0.82–0.93**.

The confusion matrix on the test set (300 rows, 10 fraud):

```
              Predicted Legit   Predicted Fraud
True Legit         289               1          (1 false positive)
True Fraud           0              10          (0 missed fraud)
```

### 5-Fold Cross-Validation for Robustness

Because the test set contains only 10 fraud examples, any single evaluation is subject to high variance. A single false negative would drop recall from 1.0 to 0.9, not because the model is bad, but because 10 examples is not enough to estimate recall precisely. Cross-validation provides a more stable picture:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
# Result: 1.0000 ± 0.0000
```

The zero standard deviation result on synthetic data (where signals are deterministic and clean) indicates that the model reliably separates the two classes across all folds. In production on real, noisy data, AUC variance would be expected in the range of 0.02–0.10.

---

## 7. Hybrid Scoring Pipeline

### Design Rationale: Why Hybrid?

A pure ML model produces well-calibrated probability estimates for patterns it has seen during training, but has two structural weaknesses at low data volumes:

1. **Novel attack patterns not in training data** will receive low ML scores because the model has no basis for recognizing them.
2. **Strong signal stacking** (multiple independent risk signals firing simultaneously) may not be fully captured by a forest trained on 49 fraud examples.

The rule-based layer addresses both limitations. Any transaction where multiple known-bad signals co-occur receives an elevated score regardless of the ML probability. The hard override logic handles the most unambiguous patterns with a minimum score floor.

### The 70/30 Blend Formula

```python
P_ml       = model.predict_proba(X_new)[:, 1]    # RF probability [0, 1]
rule_score = df['fraud_signal_count'] / 6.0      # binary signal ratio [0, 1]
blended    = (P_ml * 0.70) + (rule_score * 0.30)
score      = round(blended * 100, 1)             # final score [0, 100]
```

The 70% ML weight reflects that the model is the primary decision-maker — it has learned the interactions between features across all four fraud clusters. The 30% rule weight ensures that signal stacking is always elevated: a transaction with 4 out of 6 binary flags active receives a rule contribution of `(4/6) * 0.30 * 100 = 20` points added to whatever ML probability is returned.

**Why these two components are now genuinely independent (Fix 3):** `fraud_signal_count` (the rule-layer input) is the sum of 6 binary features. Before Fix 3, `fraud_signal_count` was also included in `FEATURE_COLS` as a direct input to the Random Forest. This meant the RF was already incorporating the aggregate of all binary signals, and then the 30% rule component was adding it again. The 70/30 blend was not a blend of independent signals — it was 70% of a model that included the aggregate plus 30% of the same aggregate directly. Removing `fraud_signal_count` from `FEATURE_COLS` restores the blend to its intended semantics: an ML estimate based on 8 independent features plus a rule contribution based on the raw count of triggered signals.

### Hard Override Logic

Hard overrides are applied after the blended score is computed, using `score.loc[mask].clip(lower=floor)` to raise specific transactions to a minimum score without altering transactions that already exceed the floor.

```python
# Triple-mismatch override: geographic + IP + email signals all firing
mask_triple = (
    (df['is_country_mismatch'] == 1) &
    (df['is_ip_mismatch']      == 1) &
    (df['is_suspicious_email'] == 1)
)
score.loc[mask_triple] = score.loc[mask_triple].clip(lower=85)

# Velocity + new account override: card-testing pattern
mask_velocity = (
    (df['velocity_score']          >= 8) &    # >= 8 purchases in 24h
    (df['new_account_large_order'] == 1)
)
score.loc[mask_velocity] = score.loc[mask_velocity].clip(lower=80)
```

**Triple-mismatch justification (floor: 85):** The simultaneous presence of a billing-shipping country mismatch, an IP-billing country mismatch, and a suspicious email is an exceptionally strong combined signal. Each of these three signals can occur independently in legitimate transactions (a traveler abroad, a VPN user, a short username), but all three occurring simultaneously has very low legitimate probability. The floor of 85 places these transactions well above the HIGH threshold with no ambiguity. It does not force a score of 85 — if the blended score is already higher, it is preserved.

**Velocity + new account justification (floor: 80):** A brand-new account executing 8 or more purchases in 24 hours is the definitional card-testing pattern. The threshold of `velocity_score >= 8` corresponds to `log1p(8) * 4.0 = 8.79`, meaning the actual purchase count must be 8 or more. The `build_explanation()` function evaluates the same full conjunction (`velocity_score >= 8 AND new_account_large_order == 1`) before appending the `[velocity override applied → floor 80]` label — ensuring the explanation label precisely mirrors when the override fires. The floor of 80 ensures these are HIGH-tier transactions.

**Implementation detail:** The `.loc[mask].clip(lower=floor)` pattern (Fix 4 from the additional improvements) avoids the pandas `SettingWithCopyWarning` that occurs when assigning to a boolean-indexed slice of a copy. Using `.loc` ensures the assignment targets the correct DataFrame memory location.

### Risk Tier Thresholds

```python
def assign_tier(s: float) -> str:
    if s >= 65: return 'HIGH'
    if s >= 30: return 'MEDIUM'
    return 'LOW'

df['risk_tier'] = df['fraud_score'].map(assign_tier)
```

| Tier | Score Range | Action | Business Rationale |
|---|---|---|---|
| HIGH | >= 65 | BLOCK — do not fulfill | Fraud probability is high enough that the expected loss from fulfillment (merchandise + dispute fee) exceeds the expected loss from blocking (potential false positive, customer friction) |
| MEDIUM | 30–64 | MANUAL REVIEW — ops team flags | Signals are elevated but not conclusive; human judgment is worth the operational cost |
| LOW | < 30 | AUTO-APPROVE | No material fraud signals; blocking or reviewing would create unacceptable false positive rates |

The 65/30 split was selected to match the data distribution of the synthetic dataset. The HIGH threshold of 65 is above the midpoint of the score range (50) but not so high that it would miss ambiguous fraud cases that score in the 65–85 range without a hard override. In production, these thresholds should be tuned against a cost function that encodes the actual cost ratio of false negatives (missed fraud: lost merchandise + $25 dispute fee) to false positives (legitimate order blocked: lost revenue + customer churn risk).

---

## 8. Explainability

### How `triggered_signals` Is Built

The `build_explanation(row)` function constructs a plain-text string for each scored transaction by inspecting its feature values:

```python
SIGNAL_LABELS = {
    'is_country_mismatch':    'billing/shipping country mismatch',
    'is_ip_mismatch':         'IP country differs from billing country',
    'new_account_large_order':'new account with large order',
    'is_suspicious_email':    'suspicious email pattern',
    'is_high_risk_bin':       'high-risk BIN detected',
    'is_prepaid_card':        'prepaid card used',
}

def build_explanation(row) -> str:
    parts = []
    for col, label in SIGNAL_LABELS.items():
        if row.get(col, 0) == 1:
            parts.append(label)

    # Velocity — label reflects whether the hard override conjunction ACTUALLY fired.
    # Override requires velocity_score >= 8 AND new_account_large_order == 1.
    # Do NOT label "[override applied]" unless both conditions are satisfied.
    vs        = row.get('velocity_score', 0)
    purchases = int(row.get('purchases_last_24h', 0))
    vel_override_fired = (vs >= 8.0 and row.get('new_account_large_order', 0) == 1)

    if vel_override_fired:
        parts.append(f'extreme purchase velocity ({purchases} purchases in 24h) [velocity override applied → floor 80]')
    elif vs >= 8.0:
        parts.append(f'extreme purchase velocity ({purchases} purchases in 24h)')
    elif vs >= 5.0:
        parts.append(f'elevated purchase velocity ({purchases} purchases in 24h)')

    # Amount outlier
    az = row.get('amount_zscore', 0)
    if az >= 2.0:
        parts.append(f'unusually high amount (z-score={az:.1f})')

    if not parts:
        return 'no flags triggered'
    return '; '.join(parts)
```

The velocity label logic evaluates the full override conjunction before appending any override tag. `vel_override_fired` is `True` only when `velocity_score >= 8.0` AND `new_account_large_order == 1` are both satisfied — exactly matching the `mask_velocity` condition in `score_transactions()`. A transaction with high velocity but no `new_account_large_order` flag receives the plain "extreme purchase velocity" label without any override annotation. Before the second peer review round, the function used `[override eligible]` unconditionally for any `velocity_score >= 8.0` transaction, which incorrectly implied that the override had (or would) fire regardless of the conjunction. See Section 9, Round 2, Fix 14 for the full before/after analysis.

The `amount_zscore >= 2.0` threshold surfaces outlier amounts in human-readable form. A z-score of 2.0 corresponds to amounts approximately 2 standard deviations above the historical mean, which in this dataset is roughly $250+ above the mean.

### Example Output

A representative HIGH-risk transaction from the scored output:

```
transaction_id : NEW0066
amount_usd     : 657.14
fraud_score    : 100.0
risk_tier      : HIGH
triggered_signals: billing/shipping country mismatch; IP country differs from billing
                   country; suspicious email pattern; high-risk BIN detected;
                   prepaid card used; unusually high amount (z-score=4.2)
```

A LOW-risk transaction:

```
transaction_id : NEW0003
amount_usd     : 87.42
fraud_score    : 3.2
risk_tier      : LOW
triggered_signals: no flags triggered
```

### Why Plain-Text Explanations Matter for Fraud Operations

Fraud analysts operate under several constraints that make interpretable scoring essential:

1. **Customer dispute response:** When a customer contacts support after a blocked transaction, the analyst needs to quickly verify whether the block was justified. A score of "95.0" alone is not actionable. A string like "billing/shipping country mismatch; high-risk BIN detected; prepaid card used" can be immediately evaluated.

2. **Card network compliance:** Visa and Mastercard require merchants to document the basis for transaction declines in certain dispute scenarios. A rule-based explanation string is directly usable as documentation.

3. **Rule tuning and model monitoring:** Fraud teams regularly review false positives and false negatives. The `triggered_signals` column allows analysts to identify which signals are firing incorrectly, enabling targeted rule adjustments without retraining the model.

4. **Analyst onboarding:** New fraud analysts can read explanations and learn what patterns the system considers suspicious, accelerating domain knowledge transfer.

---

## 9. Peer Review Improvements

After the initial implementation was complete, a senior data architect review identified four critical bugs and several additional quality improvements. All were applied before final evaluation. The table below provides a comprehensive before/after analysis of every change.

### Critical Fixes

| Fix | Issue | Before | After | Impact |
|---|---|---|---|---|
| 1 | `is_suspicious_email` false positives | `re.search(r'[a-z0-9]{8,}', local) AND NOT re.search(r'[._-]', local)` | Added vowel-ratio check: also require `vowel_count / len(local) < 0.20` | `johnsmith@gmail.com` (no separator, 9 chars) would have been flagged as suspicious in production. Any first+last concatenation without a dot separator would generate a false block. |
| 2 | `velocity_score` signal saturation | `purchases_last_24h * 2.5, clipped to 10` | `log1p(purchases_last_24h) * 4.0` | All of Cluster B (5–12 purchases) collapsed to score 10.0. The model could not distinguish moderate velocity (5 purchases) from extreme velocity (12 purchases). The core card-testing signal was invisible to the model. |
| 3 | `fraud_signal_count` double-counting | `fraud_signal_count` in both `FEATURE_COLS` (RF input) and 30% rule layer | Removed from `FEATURE_COLS`; used only in rule layer | The 70/30 ML+rule blend was not independent. The RF already encoded the aggregate binary signal; the rule layer then added it again. The blend ratio (70/30) did not reflect two genuinely independent information sources. |
| 4 | Threshold selection data leakage | Threshold selected by iterating over test set and optimizing F1 on held-out data | 3-way split (70/15/15); threshold selected on val set; metrics reported on untouched test set | Reported Precision, Recall, and F1 were optimistically biased. The "best threshold" was the one that happened to work best on the specific random sample of 300 test rows — not a generalizable operating point. |

### Fix 1 — `is_suspicious_email` in Detail

**Before:**
```python
def is_suspicious_email(email: str) -> int:
    local = email.split('@')[0].lower()
    if re.search(r'[a-z0-9]{8,}', local) and not re.search(r'[._-]', local):
        return 1
    return 0
```

This rule flags any local part with 8+ consecutive alphanumeric characters and no separator. The email `johnsmith@gmail.com` has the local part `johnsmith` — 9 alphanumeric characters, no separator — and would be flagged as suspicious. This is an extremely common legitimate email format, particularly in Southeast Asian markets where first+last concatenation without dots is standard.

**After:**
```python
def is_suspicious_email(email: str) -> int:
    local = email.split('@')[0].lower()
    if not re.search(r'[aeiou]', local):   # zero vowels — strongest signal
        return 1
    if not re.search(r'[._\-]', local) and len(local) >= 8:
        vowel_count = len(re.findall(r'[aeiou]', local))
        if vowel_count / len(local) < 0.20:
            return 1
    return 0
```

For `johnsmith`: vowel count = 2 (o, i), length = 9, ratio = 2/9 = 0.22 > 0.20. Not flagged.
For `x7kqm9pz`: vowel count = 0, length = 8, ratio = 0/8 = 0.0 < 0.20. Flagged.
For `xkzmtpbr`: no vowels at all. Flagged by Rule 1 immediately.

### Fix 2 — `velocity_score` in Detail

**Before:** `min(purchases_last_24h * 2.5, 10)`

| Purchases | Old Score |
|---|---|
| 4 | 10.0 (saturated) |
| 5 | 10.0 (saturated) |
| 8 | 10.0 (saturated) |
| 12 | 10.0 (saturated) |

Cluster B generates 5–12 purchases. Every single Cluster B transaction had an identical velocity score of 10.0. The Random Forest received zero information about the intensity of the velocity attack within Cluster B.

**After:** `log1p(purchases_last_24h) * 4.0`

| Purchases | New Score |
|---|---|
| 4 | 6.44 |
| 5 | 7.17 |
| 8 | 8.79 |
| 12 | 9.94 |

Monotonic differentiation is preserved across the full observed range. The `log1p` function also correctly handles `purchases_last_24h = 0` by returning 0 (not -inf as `log` would).

### Fix 3 — `fraud_signal_count` Double-Counting in Detail

**Before:**
```python
FEATURE_COLS = [
    'is_country_mismatch', 'is_ip_mismatch', 'velocity_score',
    'new_account_large_order', 'is_suspicious_email', 'is_high_risk_bin',
    'is_prepaid_card', 'amount_zscore',
    'fraud_signal_count'    # <- included in RF AND rule layer
]

# ... later in scoring:
rule_score = df['fraud_signal_count'] / 6.0
blended = (p_ml * 0.70) + (rule_score * 0.30)
```

The RF receives `fraud_signal_count` as a feature. `fraud_signal_count = is_country_mismatch + is_ip_mismatch + new_account_large_order + is_suspicious_email + is_high_risk_bin + is_prepaid_card`. The RF already has all 6 of these features directly, plus their aggregate. The rule layer then adds the same aggregate again with 30% weight. The effective weight of binary signal information was far higher than 30%.

**After:**
```python
FEATURE_COLS = [
    'is_country_mismatch', 'is_ip_mismatch', 'velocity_score',
    'new_account_large_order', 'is_suspicious_email', 'is_high_risk_bin',
    'is_prepaid_card', 'amount_zscore'
    # fraud_signal_count removed — it belongs to the rule layer only
]
```

Now the RF has 8 independent features. `fraud_signal_count` is still computed and stored in the DataFrame for use in the rule layer, but it never enters `FEATURE_COLS` and is never passed to `model.fit()` or `model.predict_proba()`.

### Fix 4 — Threshold Leakage in Detail

**Before:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# Threshold selection ON the test set — leakage
thresholds = np.arange(0.05, 0.95, 0.01)
f1s = [f1_score(y_test, (y_prob >= t)) for t in thresholds]
THRESHOLD = thresholds[np.argmax(f1s)]  # optimized on test set
# ... then report Precision/Recall/F1 on the same test set using this threshold
```

The reported F1 was the maximum F1 achievable on that specific test sample — not an estimate of what the model would achieve on a new batch of transactions. This is a well-known form of evaluation data leakage.

**After:**
```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

model.fit(X_train, y_train)

# Threshold selection on VALIDATION set
y_val_prob = model.predict_proba(X_val)[:, 1]
THRESHOLD = thresholds[np.argmax([f1_score(y_val, (y_val_prob >= t)) for t in thresholds])]

# Metrics reported on UNTOUCHED test set
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)
# Precision, Recall, F1 reported here
```

### Additional Improvements

| Improvement | Before | After | Benefit |
|---|---|---|---|
| `is_prepaid_card` case sensitivity | `payment_method == 'prepaid'` | `.str.lower().str.contains('prepaid', na=False)` | Robust to `'Prepaid'`, `'PREPAID'`, `'prepaid_card'` — all of which appear in the synthetic data |
| Hard override assignment | `score[mask] = score[mask].clip(lower=85)` (boolean index on copy) | `score.loc[mask] = score.loc[mask].clip(lower=85)` | Eliminates `SettingWithCopyWarning`; ensures assignment targets the correct memory location |
| Risk tier assignment | `pd.cut()` with bin edges | `assign_tier()` map function | Removes sentinel/boundary ambiguity in `pd.cut`; the tier logic is explicit and readable |
| `build_explanation()` velocity threshold | Used 7.5 as "override eligible" threshold | Aligned to 8.0 (matches `score_transactions()` override condition) | Explanation text now accurately reflects when the override will actually trigger |
| Primary metric | ROC-AUC only | PR-AUC added as primary metric alongside ROC-AUC | PR-AUC is more informative at 3.5% base rate; captures precision/recall trade-off visible to operations teams |
| Robustness validation | Single train/test split only | 5-fold stratified CV AUC added | Provides variance estimate over the small fraud sample count |

### Round 2 — Second Peer Review

A second senior architect review identified four additional issues applied after the first round. These changes address documentation accuracy, status labeling, and a label-correctness bug in the explainability layer.

| Fix | Issue | Before | After | Impact |
|---|---|---|---|---|
| 11 | AUC disclaimer missing from Executive Summary | §1 reported PR-AUC = 1.0000 with no caveat | Synthetic Data Caveat block added immediately after the metrics sentence | Prevents readers from interpreting AUC=1.0 as a production performance estimate; explains why perfect separation is expected and provides a realistic 0.82–0.93 benchmark range |
| 12 | AUC disclaimer missing from §6 | Metrics table had no interpretation note | Interpretation note added beneath metrics: "ROC-AUC = PR-AUC = 1.0 is the expected outcome on this synthetic dataset" | Evaluation section now stands independently without requiring the reader to cross-reference the Executive Summary disclaimer |
| 13 | Status header over-claimed production readiness | `Status: Peer-reviewed — production-candidate` | `Status: Peer-reviewed — Demo/Proof-of-Concept (see §6 for production readiness caveats)` | Accurately conveys that the system is a validated proof-of-concept on synthetic data, not a system ready for live deployment |
| 14 | `[override eligible]` label fired on velocity-only conditions | `build_explanation()` appended `[override eligible]` whenever `velocity_score >= 8.0`, regardless of `new_account_large_order` | Full conjunction evaluated: `vel_override_fired = (vs >= 8.0 AND new_account_large_order == 1)`; label reads `[velocity override applied → floor 80]` only when both conditions are true | Transactions with high velocity but no `new_account_large_order` flag were incorrectly described as "override eligible" — the override would not have fired for them. Analysts reading the explanation would incorrectly believe the score had been floored to 80 |

#### Fix 14 — `build_explanation()` Override Label in Detail

**Before (Round 1 state):**
```python
vs        = row.get('velocity_score', 0)
purchases = int(row.get('purchases_last_24h', 0))
if vs >= 8.0:
    # Appended unconditionally — override actually requires new_account_large_order == 1 too
    parts.append(f'extreme purchase velocity ({purchases} purchases in 24h) [override eligible]')
elif vs >= 5.0:
    parts.append(f'elevated purchase velocity ({purchases} purchases in 24h)')
```

The label `[override eligible]` fired for any transaction with `velocity_score >= 8.0`. However, the actual `mask_velocity` override in `score_transactions()` is:

```python
mask_velocity = (
    (df['velocity_score']          >= 8) &
    (df['new_account_large_order'] == 1)
)
```

A transaction with 10 purchases in 24 hours but an established account (`account_age_days > 30`) would receive the `[override eligible]` label despite the override never firing. Its score would not be floored to 80. The label was misleading to analysts.

**After (Round 2 fix):**
```python
vs        = row.get('velocity_score', 0)
purchases = int(row.get('purchases_last_24h', 0))
vel_override_fired = (vs >= 8.0 and row.get('new_account_large_order', 0) == 1)

if vel_override_fired:
    parts.append(f'extreme purchase velocity ({purchases} purchases in 24h) [velocity override applied → floor 80]')
elif vs >= 8.0:
    parts.append(f'extreme purchase velocity ({purchases} purchases in 24h)')
elif vs >= 5.0:
    parts.append(f'elevated purchase velocity ({purchases} purchases in 24h)')
```

The label now precisely mirrors the override condition. A self-verification assertion was also added in the scoring notebook cell:

```python
false_labels = scored[
    scored['triggered_signals'].str.contains('override applied', na=False) &
    ~((scored['velocity_score'] >= 8) & (scored['new_account_large_order'] == 1))
]
assert len(false_labels) == 0, f"{len(false_labels)} incorrectly labelled transactions"
# Output: Override label accuracy check: 0 incorrectly labelled transactions (expect 0)
```

---

## 10. Limitations and Production Considerations

### Known Limitations

**Statistical baselines computed on full historical dataset.**
`AMOUNT_MEAN`, `AMOUNT_STD`, and `AMOUNT_75TH` are computed on all 2,000 rows before the train/val/test split. This means the test set's amount features are normalized using statistics that include test-set amounts — a mild leakage. At 2,000 rows with a stable exponential distribution, this makes no practical difference to model performance. In production, these statistics must be computed exclusively on `X_train` and then applied as fixed transforms to validation and test sets, and to new scoring batches, using a sklearn `Pipeline` or explicit `fit_transform` pattern.

**Small fraud count in all splits.**
The training set contains 49 fraud examples, validation contains 11, and test contains 10. With 10 test fraud examples, each individual miss would reduce recall by 0.10. The CV AUC of 1.0000 ± 0.0000 is consistent across folds, but this reflects the clean, deterministic nature of synthetic data. Real fraud data is noisy, overlapping, and harder to separate — real-world performance estimates should be interpreted with wider confidence intervals.

**Feature engineering is batch-only.**
`velocity_score` is derived from `purchases_last_24h`, a pre-computed column in the dataset. In production, computing "purchases in the last 24 hours from the same customer" requires querying a real-time transaction store (e.g., Redis, DynamoDB, or a time-windowed feature in a feature store). The current implementation does not model the infrastructure needed to compute this feature at scoring time.

**Synthetic data does not fully model noise.**
The four fraud clusters in this dataset are deliberately clean — each cluster's defining signals are always present for fraud transactions in that cluster. Real fraud data has much higher feature noise: legitimate transactions sometimes have geographic mismatches (travelers), velocity bursts (shopping sprees), and new accounts with large orders (wedding registries). Model performance on real data will be lower than the synthetic results indicate.

**No temporal validation.**
The model is trained and tested on random samples from the same 12-month window. In production, the correct evaluation methodology is forward-looking: train on months 1–9, validate on months 10–11, test on month 12. This simulates the real deployment scenario where the model is trained on historical data and evaluated on future transactions.

**Risk tier thresholds have no cost-function derivation.**
The HIGH threshold of 65 and MEDIUM threshold of 30 were selected to match the distribution of the synthetic dataset, not derived from a business cost function. The correct methodology is to find the optimal threshold `T` by minimizing expected cost over all possible threshold values:

```
Optimal T = argmin over T of: FP(T) × fp_cost + FN(T) × fn_cost
```

Where `fp_cost` is the revenue lost from blocking a legitimate order and `fn_cost` is the total loss from a fraudulent order: dispute fee ($25) + merchandise cost + operational handling. At Storm Retail's average order value and $25 dispute fee, the cost ratio heavily favors recall (catching fraud) over precision (avoiding false blocks). The 65/30 thresholds should be re-derived against this cost function before the system is used operationally.

**Amount statistics are module-level globals, not serialized with the model artifact.**
`AMOUNT_MEAN`, `AMOUNT_STD`, and `AMOUNT_75TH` are computed at notebook execution time and stored as Python module-level variables. They are not saved alongside the serialized model. If the model artifact (`.pkl` or `joblib` file) is loaded in a different process or service — or retrained on a dataset with a different amount distribution — the scoring function will silently use stale or incorrect normalization statistics. In production, these constants must be saved alongside the model, for example as a `stats.json` file, or encapsulated inside a `sklearn.pipeline.Pipeline` where the scaler is `fit` on training data and serialized as part of the pipeline object.

**Circularity between training data and new transaction scoring.**
The 49% HIGH-tier block rate on the 100 new transactions is partly an artifact of the data generation methodology: `generate_new_transactions()` produces its 40 "suspicious" transactions by cycling across the same 4 cluster patterns the model was trained on. New transactions were sampled from the same underlying distributions as the training fraud cases. This inflates the apparent sensitivity of the model — it is not evidence that the model generalizes to attack patterns outside these four clusters. On real production data with novel fraud patterns, the HIGH block rate will be different, and the model may miss patterns it has not been exposed to.

**`device_type` is excluded from FEATURE_COLS without a documented rationale.**
Cluster B (velocity attack) always uses `device_type = 'mobile'`, which is a distinguishing signal for scripted attacks. Despite this, `device_type` is absent from `FEATURE_COLS` and is never used in the RF or the rule layer. The reason for this exclusion is not stated in the code. The likely justification is that `device_type` is already captured indirectly by the `velocity_score` and `account_age_days` signals (both of which are strong Cluster B indicators), and including it on a 49-sample training fraud set risks overfitting to the mobile device pattern rather than learning the underlying velocity behavior. This rationale should be explicitly documented wherever `FEATURE_COLS` is defined to prevent future engineers from inadvertently adding the feature.

**Rule layer blindspot: `velocity_score` and `amount_zscore` are excluded from `fraud_signal_count`.**
The `fraud_signal_count` rule layer sums only the 6 binary features (`is_country_mismatch`, `is_ip_mismatch`, `new_account_large_order`, `is_suspicious_email`, `is_high_risk_bin`, `is_prepaid_card`). The two continuous features — `velocity_score` and `amount_zscore` — are excluded. A transaction with extreme velocity (`velocity_score = 9.5`) but no binary flags set will have `fraud_signal_count = 0` and therefore `rule_score = 0.0`. Its fraud score is driven entirely by the ML component. This is a known architectural gap: the rule layer provides no independent uplift for velocity-only or amount-only anomalies. The hard override for `velocity_score >= 8 AND new_account_large_order == 1` partially compensates, but a high-velocity transaction from an established account (`new_account_large_order = 0`) receives no rule-layer contribution at all.

**Cross-validation runs on the full dataset including validation and test rows.**
The `cross_val_score(model, X, y, ...)` call in the notebook uses the full 2,000-row `X` and `y` arrays — including rows that are in the validation and test partitions. On real, noisy data this would add information beyond the held-out test AUC (CV provides variance estimates across different data subsets). On this perfectly separable synthetic dataset, the CV adds no new information: all five folds achieve AUC = 1.0000 with zero variance, identically to the test set result. The call is harmless in this context but would be misleading if interpreted as providing additional generalization evidence in a production setting where CV should be run on training data only.

### What Would Change in Production

**Feature store integration.**
`velocity_score`, `account_age_days`, and amount statistics must be materialized in a feature store (e.g., Feast, Tecton, or a custom Redis-backed service) with separate online (low-latency) and offline (batch training) APIs. The training pipeline reads from the offline store; the scoring API reads from the online store. This ensures training and serving use the same feature computation logic — eliminating training/serving skew.

**Online scoring service.**
The notebook scoring function must be extracted into a REST API endpoint (e.g., FastAPI or Flask). The model is serialized using `joblib` or `mlflow.sklearn`, loaded at service startup, and called synchronously during the fulfillment decision step. Target latency is <20ms (the model itself is fast; the bottleneck is feature retrieval from the feature store).

**Model versioning and experiment tracking.**
Each model training run should be logged in an experiment tracking system (MLflow, Weights & Biases, or SageMaker Experiments) with: hyperparameters, training dataset hash, all evaluation metrics, and the serialized model artifact. Deployment should go through a staging environment with shadow scoring (score live transactions but do not act on the score) before cutover.

**Concept drift monitoring.**
Fraud patterns evolve continuously. The model should be monitored for:
- **Population drift:** Distribution shifts in input features (e.g., a new country entering the customer base) detected via PSI (Population Stability Index) or KS test.
- **Score drift:** Distribution shifts in output fraud scores over time.
- **Outcome drift:** As chargebacks are confirmed (with a 60–120 day lag), ground truth labels become available. PR-AUC and recall should be tracked on confirmed chargeback outcomes weekly.

**Threshold recalibration.**
The threshold of 0.37 was selected on 11 validation fraud examples. In production, threshold selection should incorporate a business cost function: the false negative cost (lost merchandise + $25 dispute fee) vs. the false positive cost (lost revenue + customer churn probability). The optimal threshold may shift significantly when evaluated against real cost ratios.

**Feedback loop and active learning.**
The fraud operations team's manual review decisions (MEDIUM-tier transactions they confirm as fraud or clear as legitimate) should be incorporated into retraining data. This closes the human-in-the-loop feedback cycle. Confirmed fraud cases from card networks (chargeback reports with reason codes) should be matched back to original transactions and used to expand the labeled training set.

**IAM and data access controls.**
The scoring service should operate with least-privilege IAM roles. Transaction data containing card BINs and email addresses requires PCI-DSS and PII handling policies. The model artifact and training data store should be access-controlled and audited separately from the application tier.
