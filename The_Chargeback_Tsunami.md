The Chargeback Tsunami: Predict Storm Retail's Fraud Wave

Scenario
Storm Retail, a fast-growing Southeast Asian fashion e-commerce platform, just received devastating news from their payment orchestration provider (Yuno): their chargeback rate has spiked from 0.4% to 2.8% over the past 45 days. They're now dangerously close to breaching the 3% threshold that would trigger their acquirers to freeze their accounts — effectively shutting down their ability to accept card payments.

The CFO is in panic mode. Chargebacks cost Storm Retail not just the lost merchandise and transaction amount, but also $25 USD in dispute fees per incident. With 50,000 transactions per month, they're hemorrhaging approximately $35,000 monthly just in fees, not counting the lost goods.

Storm Retail's fraud prevention team has been operating reactively — manually reviewing suspicious orders after they ship. They need to shift to a predictive approach: identify high-risk transactions in real-time before fulfillment, so they can block or manually review suspicious orders before shipping products that will never be paid for.

Yuno's client success team has escalated this to engineering. Storm Retail needs a working proof-of-concept fraud scoring system that can analyze historical transaction data, identify patterns associated with chargebacks, and predict which future transactions are likely to result in disputes.

Domain Background
Before diving into the challenge, here are the key payment concepts you need to understand:

What is a Chargeback?
A chargeback occurs when a cardholder disputes a transaction with their card issuer (the bank that issued their credit/debit card) and the funds are forcibly reversed from the merchant. Common reasons include:

Fraud: The cardholder didn't make the purchase (stolen card)
Product not received: Customer claims they never got the item
Product not as described: Item significantly differs from listing
Duplicate charge: Customer was charged multiple times
Authorization issues: Merchant charged without proper authorization
Chargebacks are expensive for merchants because they lose both the product (already shipped) and the transaction amount, plus dispute fees ($15-$100 per chargeback).

What is a Chargeback Rate?
Chargeback rate = (Number of chargebacks / Total number of transactions) × 100

Card networks (Visa, Mastercard) enforce strict limits. If a merchant's chargeback rate exceeds 1% for two consecutive months, they enter monitoring programs with fines. Above 3%, acquirers may terminate the merchant relationship entirely.

What is an Acquirer?
An acquirer (or acquiring bank) is the financial institution that processes card payments on behalf of merchants. They take on financial risk and enforce compliance with card network rules. If a merchant becomes too risky (high chargebacks), the acquirer will drop them.

What is BIN?
BIN (Bank Identification Number), also called IIN, is the first 6-8 digits of a card number. It identifies the issuing bank and card type. Certain BINs are associated with higher fraud rates (e.g., prepaid cards, cards from specific regions).

Transaction Lifecycle
Authorization: Customer enters card details; issuer approves or declines in real-time
Capture: Merchant captures the authorized amount (usually immediately for e-commerce)
Fulfillment: Merchant ships the product (this is where fraud detection matters most!)
Settlement: Funds move from issuer → card network → acquirer → merchant (1-3 days)
Chargeback window: Cardholder has 60-120 days to dispute the charge
Your Mission
Build a fraud prediction system that analyzes Storm Retail's historical transaction data to identify patterns that precede chargebacks, then scores new transactions on their fraud risk.

Your solution should consist of:

A data pipeline that ingests and processes transaction history
A predictive model or scoring algorithm that identifies high-risk transactions
A visual analytics component that helps Storm Retail's fraud team understand patterns, evaluate model performance, and review flagged transactions
You may use any programming language, frameworks, libraries, or approach you prefer. The solution should be demonstrable — a reviewer should be able to run it, see insights from historical data, and score new transactions.

Test Data Specification
You will need to generate or synthesize transaction data for development and demonstration. Your test dataset should include:

Historical Transactions (for model training/analysis):

At least 2,000 transactions spanning 90 days
Transaction attributes should include: transaction ID, timestamp, amount (USD), customer email domain, customer country, shipping country, BIN (first 6 digits of card), payment method (credit/debit/prepaid), time between account creation and purchase, order velocity (purchases in last 24h from same customer), product category, device type (mobile/desktop), IP country
Approximately 2-5% of transactions should be labeled as chargebacks (these represent confirmed fraud or disputes)
Include realistic patterns that correlate with fraud, such as:
Mismatched billing/shipping countries
High velocity (multiple purchases from same email in short time)
Unusually large order amounts for new accounts
Suspicious email patterns (e.g., randomly generated emails)
Specific high-risk BINs or countries
New Transactions (for scoring):

At least 100 unlabeled transactions representing today's orders
Should include a mix of clearly safe, clearly suspicious, and ambiguous cases
Should contain enough suspicious signals to demonstrate the model's predictive capability
You may generate this data programmatically, use synthetic data libraries, or leverage AI tools. The data does not need to be perfectly realistic — it should be sufficient to demonstrate your solution's capabilities.

Functional Requirements
Core Requirements (Must Complete)
1. Pattern Analysis & Feature Engineering

Ingest the historical transaction dataset
Analyze which transaction characteristics are most strongly associated with chargebacks
Engineer at least 5 meaningful features/signals that could predict fraud risk (e.g., velocity metrics, geographic mismatches, account age, order size anomalies)
Document which patterns you discovered and why they matter
2. Fraud Scoring System

Build a system that assigns a fraud risk score (0-100 or 0-1) to any transaction
Your approach can be rule-based, ML-based, statistical, or hybrid — choose what you can deliver well in 2 hours
Score the 100 new transactions in your test set
Classify transactions into risk tiers: LOW (safe to auto-approve), MEDIUM (manual review), HIGH (block or reject)
The system should output scored transactions with explanations (which signals triggered the score)
3. Visual Analytics Dashboard

Create a visual interface (web-based, notebook, or desktop app) that displays:
Model performance metrics: If using ML, show precision, recall, confusion matrix, or ROC curve on historical data
Top risk factors: Which features most strongly predict chargebacks?
Flagged transactions: A sortable view of the 100 newly scored transactions, highlighting high-risk cases
At least 2 additional visualizations that help the fraud team understand patterns (e.g., chargeback rate by country, velocity distribution, BIN analysis, time-series trends)
Stretch Goals (Optional — Partial Completion Expected)
4. Real-Time Simulation

Create a simple mechanism to simulate processing transactions in real-time (e.g., read from a queue, API endpoint, or streaming file)
Show how the system would score transactions as they arrive
5. Business Impact Calculator

Estimate cost savings: Given a decision threshold, calculate how much Storm Retail would save by blocking high-risk transactions vs. the revenue they'd lose from false positives
Allow the fraud team to tune the risk threshold and see the trade-off
Partial completion of stretch goals is expected and welcomed. Focus on delivering a strong core solution first.

Acceptance Criteria
A complete submission should demonstrate:

✅ Runnable system: A reviewer can execute your solution (with clear README instructions) and see results
✅ Data-driven insights: Clear analysis of what patterns predict chargebacks in your historical data
✅ Working fraud scoring: The 100 new transactions are scored, and high-risk cases are clearly identified
✅ Useful visualizations: The fraud team can explore patterns and understand model decisions
✅ Explainability: For flagged transactions, it's clear WHY they were scored as risky
✅ Documentation: README explains your approach, how to run the system, and key findings

Deliverables
At the end of the challenge, submit:

Source code (all scripts, notebooks, or application code)
README.md with setup instructions, dependencies, how to run the solution, and a summary of your approach
Generated test data files (or script to generate them)
Output artifacts: scored transactions, visualizations, or screenshots of the dashboard
(Optional) A brief video or loom walkthrough (2-3 minutes) demonstrating your solution in action
Evaluation Notes
You will NOT be judged on:

Using a specific ML algorithm or framework
Production-readiness, scalability, or deployment infrastructure
Perfectly realistic data
You WILL be judged on:

Delivering a working, demonstrable solution in 2 hours
Analytical rigor: thoughtful feature engineering and pattern discovery
Clarity: clean code, good documentation, understandable visualizations
Business value: does this actually help Storm Retail prevent chargebacks?
Good luck — Storm Retail is counting on you! 🚀



                                                                                                                                                                 