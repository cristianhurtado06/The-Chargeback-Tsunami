"""
Synthetic transaction data generator for Chargeback Tsunami challenge.
Generates 2,000 historical transactions with 4 embedded fraud clusters
and 100 unlabeled new transactions for scoring.
"""

import numpy as np
import pandas as pd
import random
import string
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────

COUNTRIES = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'ES', 'IT', 'BR', 'MX',
             'CO', 'AR', 'CL', 'PE', 'EC', 'NG', 'RO', 'BD', 'PK', 'UA']
HIGH_RISK_COUNTRIES = ['NG', 'RO', 'BD', 'PK', 'UA']
LOW_RISK_COUNTRIES  = ['US', 'GB', 'CA', 'AU', 'DE', 'FR']

PAYMENT_METHODS = ['credit_card', 'debit_card', 'prepaid', 'paypal', 'bank_transfer']
PRODUCT_CATEGORIES = ['electronics', 'clothing', 'jewelry', 'prepaid_card',
                      'home_appliances', 'books', 'sporting_goods', 'beauty']
DEVICE_TYPES = ['mobile', 'desktop', 'tablet']

HIGH_RISK_BINS = [412345, 511234, 601100, 372345, 349876]
NORMAL_BINS    = [411111, 424242, 532532, 601611, 373737, 448100, 540540, 601109]

EMAIL_DOMAINS = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                 'icloud.com', 'protonmail.com', 'mail.com', 'gmx.com']


# ── Helpers ────────────────────────────────────────────────────────────────────

def random_email(suspicious: bool = False) -> str:
    if suspicious:
        # 8+ random alphanumeric chars, no vowels
        chars = string.ascii_lowercase.replace('a','').replace('e','').replace('i','')\
                                      .replace('o','').replace('u','') + string.digits
        local = ''.join(random.choices(chars, k=random.randint(8, 14)))
    else:
        first = random.choice(['john','jane','mike','sarah','alex','chris','emma','david',
                               'lisa','tom','anna','peter','maria','luis','ana'])
        last  = random.choice(['smith','jones','garcia','rodriguez','johnson','lee',
                               'brown','davis','wilson','martinez'])
        local = f"{first}.{last}{random.randint(1,999)}"
    domain = random.choice(EMAIL_DOMAINS)
    return f"{local}@{domain}"

def random_bin(high_risk: bool = False) -> int:
    if high_risk:
        return random.choice(HIGH_RISK_BINS)
    return random.choice(NORMAL_BINS)

def random_timestamp(start: str = '2024-01-01', end: str = '2024-12-31') -> str:
    start_ts = pd.Timestamp(start).value // 10**9
    end_ts   = pd.Timestamp(end).value   // 10**9
    ts = int(rng.integers(start_ts, end_ts))
    return pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')


# ── Cluster builders ───────────────────────────────────────────────────────────

def build_cluster_a(n: int) -> list[dict]:
    """Billing/shipping country mismatch + high-risk destination."""
    rows = []
    for i in range(n):
        billing = random.choice(LOW_RISK_COUNTRIES)
        shipping = random.choice(HIGH_RISK_COUNTRIES)
        ip = random.choice(HIGH_RISK_COUNTRIES + LOW_RISK_COUNTRIES[:3])
        amount = round(float(rng.uniform(150, 800)), 2)
        rows.append({
            'billing_country':  billing,
            'shipping_country': shipping,
            'ip_country':       ip,
            'amount_usd':       amount,
            'card_bin':         random_bin(high_risk=rng.random() > 0.5),
            'payment_method':   random.choice(['credit_card', 'prepaid']),
            'account_age_days': int(rng.integers(30, 730)),
            'purchases_last_24h': int(rng.integers(1, 4)),
            'product_category': random.choice(['electronics', 'jewelry', 'prepaid_card']),
            'device_type':      random.choice(DEVICE_TYPES),
            'customer_email':   random_email(suspicious=False),
            'is_chargeback':    1,
        })
    return rows


def build_cluster_b(n: int) -> list[dict]:
    """Velocity attack: same email many times in 4h, new account (<3 days)."""
    rows = []
    for i in range(n):
        country = random.choice(LOW_RISK_COUNTRIES + ['MX', 'BR', 'CO'])
        rows.append({
            'billing_country':  country,
            'shipping_country': country,
            'ip_country':       country,
            'amount_usd':       round(float(rng.uniform(50, 300)), 2),
            'card_bin':         random_bin(high_risk=rng.random() > 0.6),
            'payment_method':   random.choice(['credit_card', 'debit_card']),
            'account_age_days': int(rng.integers(0, 3)),
            'purchases_last_24h': int(rng.integers(5, 12)),
            'product_category': random.choice(['electronics', 'prepaid_card', 'clothing']),
            'device_type':      'mobile',
            'customer_email':   random_email(suspicious=False),
            'is_chargeback':    1,
        })
    return rows


def build_cluster_c(n: int) -> list[dict]:
    """New account (<7 days) + high amount (>$400) + electronics/prepaid."""
    rows = []
    for i in range(n):
        country = random.choice(LOW_RISK_COUNTRIES + ['MX', 'BR'])
        rows.append({
            'billing_country':  country,
            'shipping_country': country,
            'ip_country':       random.choice(HIGH_RISK_COUNTRIES + [country]),
            'amount_usd':       round(float(rng.uniform(400, 1200)), 2),
            'card_bin':         random_bin(high_risk=rng.random() > 0.4),
            'payment_method':   random.choice(['credit_card', 'prepaid']),
            'account_age_days': int(rng.integers(0, 7)),
            'purchases_last_24h': int(rng.integers(1, 5)),
            'product_category': random.choice(['electronics', 'prepaid_card']),
            'device_type':      random.choice(DEVICE_TYPES),
            'customer_email':   random_email(suspicious=rng.random() > 0.5),
            'is_chargeback':    1,
        })
    return rows


def build_cluster_d(n: int) -> list[dict]:
    """Suspicious email + high-risk BIN + high-risk IP country."""
    rows = []
    for i in range(n):
        billing = random.choice(LOW_RISK_COUNTRIES)
        rows.append({
            'billing_country':  billing,
            'shipping_country': billing,
            'ip_country':       random.choice(HIGH_RISK_COUNTRIES),
            'amount_usd':       round(float(rng.uniform(100, 600)), 2),
            'card_bin':         random_bin(high_risk=True),
            'payment_method':   random.choice(['credit_card', 'prepaid']),
            'account_age_days': int(rng.integers(5, 365)),
            'purchases_last_24h': int(rng.integers(1, 4)),
            'product_category': random.choice(PRODUCT_CATEGORIES),
            'device_type':      random.choice(DEVICE_TYPES),
            'customer_email':   random_email(suspicious=True),
            'is_chargeback':    1,
        })
    return rows


def build_legit(n: int) -> list[dict]:
    """Normal legitimate transactions."""
    rows = []
    for i in range(n):
        country = random.choice(LOW_RISK_COUNTRIES + ['MX', 'BR', 'CO', 'AR'])
        rows.append({
            'billing_country':  country,
            'shipping_country': country,
            'ip_country':       country,
            'amount_usd':       round(float(rng.exponential(80)) + 10, 2),
            'card_bin':         random_bin(high_risk=False),
            'payment_method':   random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer']),
            'account_age_days': int(rng.integers(30, 2000)),
            'purchases_last_24h': int(rng.integers(0, 3)),
            'product_category': random.choice(PRODUCT_CATEGORIES),
            'device_type':      random.choice(DEVICE_TYPES),
            'customer_email':   random_email(suspicious=False),
            'is_chargeback':    0,
        })
    return rows


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_historical(n_total: int = 2000, chargeback_rate: float = 0.035) -> pd.DataFrame:
    n_fraud = int(n_total * chargeback_rate)  # ~70
    n_legit = n_total - n_fraud

    # Distribute fraud across 4 clusters: 35%, 25%, 20%, 20%
    c_a = int(n_fraud * 0.35)
    c_b = int(n_fraud * 0.25)
    c_c = int(n_fraud * 0.20)
    c_d = n_fraud - c_a - c_b - c_c  # remainder = 20%

    fraud_rows = (build_cluster_a(c_a) + build_cluster_b(c_b) +
                  build_cluster_c(c_c) + build_cluster_d(c_d))
    legit_rows = build_legit(n_legit)
    all_rows   = fraud_rows + legit_rows

    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)
    df.insert(0, 'transaction_id', [f'TXN{i:06d}' for i in range(len(df))])
    df.insert(1, 'timestamp', [random_timestamp() for _ in range(len(df))])

    # Derive email_domain
    df['email_domain'] = df['customer_email'].str.split('@').str[1]

    # Reorder columns
    cols = ['transaction_id', 'timestamp', 'amount_usd', 'customer_email', 'email_domain',
            'billing_country', 'shipping_country', 'ip_country', 'card_bin', 'payment_method',
            'account_age_days', 'purchases_last_24h', 'product_category', 'device_type',
            'is_chargeback']
    return df[cols]


def generate_new_transactions(n: int = 100) -> pd.DataFrame:
    """
    40 clearly safe, 40 clearly suspicious, 20 ambiguous.
    No is_chargeback column.
    """
    rows = []

    # 40 safe
    for _ in range(40):
        country = random.choice(LOW_RISK_COUNTRIES)
        rows.append({
            'billing_country':  country,
            'shipping_country': country,
            'ip_country':       country,
            'amount_usd':       round(float(rng.uniform(20, 200)), 2),
            'card_bin':         random_bin(high_risk=False),
            'payment_method':   random.choice(['credit_card', 'debit_card', 'paypal']),
            'account_age_days': int(rng.integers(180, 2000)),
            'purchases_last_24h': int(rng.integers(0, 2)),
            'product_category': random.choice(['books', 'clothing', 'sporting_goods', 'beauty']),
            'device_type':      random.choice(DEVICE_TYPES),
            'customer_email':   random_email(suspicious=False),
        })

    # 40 suspicious (mix of clusters)
    for i in range(40):
        pattern = i % 4
        if pattern == 0:  # country mismatch
            billing  = random.choice(LOW_RISK_COUNTRIES)
            shipping = random.choice(HIGH_RISK_COUNTRIES)
            ip       = random.choice(HIGH_RISK_COUNTRIES)
            rows.append({
                'billing_country':  billing,
                'shipping_country': shipping,
                'ip_country':       ip,
                'amount_usd':       round(float(rng.uniform(200, 900)), 2),
                'card_bin':         random_bin(high_risk=True),
                'payment_method':   'prepaid',
                'account_age_days': int(rng.integers(10, 200)),
                'purchases_last_24h': int(rng.integers(1, 4)),
                'product_category': 'electronics',
                'device_type':      'mobile',
                'customer_email':   random_email(suspicious=True),
            })
        elif pattern == 1:  # velocity
            country = random.choice(LOW_RISK_COUNTRIES)
            rows.append({
                'billing_country':  country,
                'shipping_country': country,
                'ip_country':       country,
                'amount_usd':       round(float(rng.uniform(50, 300)), 2),
                'card_bin':         random_bin(high_risk=rng.random() > 0.5),
                'payment_method':   'credit_card',
                'account_age_days': int(rng.integers(0, 2)),
                'purchases_last_24h': int(rng.integers(6, 12)),
                'product_category': 'prepaid_card',
                'device_type':      'mobile',
                'customer_email':   random_email(suspicious=False),
            })
        elif pattern == 2:  # new account + high amount
            country = random.choice(LOW_RISK_COUNTRIES)
            rows.append({
                'billing_country':  country,
                'shipping_country': country,
                'ip_country':       random.choice(HIGH_RISK_COUNTRIES),
                'amount_usd':       round(float(rng.uniform(500, 1500)), 2),
                'card_bin':         random_bin(high_risk=rng.random() > 0.4),
                'payment_method':   random.choice(['credit_card', 'prepaid']),
                'account_age_days': int(rng.integers(0, 6)),
                'purchases_last_24h': int(rng.integers(2, 6)),
                'product_category': 'electronics',
                'device_type':      random.choice(DEVICE_TYPES),
                'customer_email':   random_email(suspicious=True),
            })
        else:  # suspicious email + high-risk BIN
            billing = random.choice(LOW_RISK_COUNTRIES)
            rows.append({
                'billing_country':  billing,
                'shipping_country': billing,
                'ip_country':       random.choice(HIGH_RISK_COUNTRIES),
                'amount_usd':       round(float(rng.uniform(100, 700)), 2),
                'card_bin':         random_bin(high_risk=True),
                'payment_method':   random.choice(['credit_card', 'prepaid']),
                'account_age_days': int(rng.integers(10, 400)),
                'purchases_last_24h': int(rng.integers(1, 5)),
                'product_category': random.choice(['jewelry', 'electronics']),
                'device_type':      random.choice(DEVICE_TYPES),
                'customer_email':   random_email(suspicious=True),
            })

    # 20 ambiguous
    for i in range(20):
        country = random.choice(LOW_RISK_COUNTRIES + HIGH_RISK_COUNTRIES[:2])
        rows.append({
            'billing_country':  country,
            'shipping_country': random.choice([country, random.choice(COUNTRIES)]),
            'ip_country':       random.choice(COUNTRIES),
            'amount_usd':       round(float(rng.uniform(100, 500)), 2),
            'card_bin':         random_bin(high_risk=rng.random() > 0.7),
            'payment_method':   random.choice(PAYMENT_METHODS),
            'account_age_days': int(rng.integers(5, 60)),
            'purchases_last_24h': int(rng.integers(1, 5)),
            'product_category': random.choice(['electronics', 'clothing', 'jewelry']),
            'device_type':      random.choice(DEVICE_TYPES),
            'customer_email':   random_email(suspicious=rng.random() > 0.6),
        })

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, 'transaction_id', [f'NEW{i:04d}' for i in range(len(df))])
    df.insert(1, 'timestamp', [random_timestamp('2025-01-01', '2025-01-31') for _ in range(len(df))])
    df['email_domain'] = df['customer_email'].str.split('@').str[1]

    cols = ['transaction_id', 'timestamp', 'amount_usd', 'customer_email', 'email_domain',
            'billing_country', 'shipping_country', 'ip_country', 'card_bin', 'payment_method',
            'account_age_days', 'purchases_last_24h', 'product_category', 'device_type']
    return df[cols]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = Path(__file__).parent

    print("Generating historical transactions...")
    hist = generate_historical(n_total=2000, chargeback_rate=0.035)
    hist_path = out_dir / 'historical_transactions.csv'
    hist.to_csv(hist_path, index=False)
    cb_rate = hist['is_chargeback'].mean() * 100
    print(f"  Saved {len(hist):,} rows → {hist_path}")
    print(f"  Chargeback rate: {cb_rate:.2f}%  ({hist['is_chargeback'].sum()} fraud / {(~hist['is_chargeback'].astype(bool)).sum()} legit)")

    print("\nGenerating new transactions...")
    new_txns = generate_new_transactions(n=100)
    new_path = out_dir / 'new_transactions.csv'
    new_txns.to_csv(new_path, index=False)
    print(f"  Saved {len(new_txns):,} rows → {new_path}")
    print("  (40 safe | 40 suspicious | 20 ambiguous — no labels)")

    print("\nDone. Ready to open the notebook.")
