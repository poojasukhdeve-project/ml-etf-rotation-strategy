# 📊 ML-Based ETF Rotation Strategy (Growth vs Value)

> *Can machine learning learn when to switch between Growth and Value stocks?*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-green.svg)](https://finance.yahoo.com/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 🧩 Background

Most investors face a simple but frustrating question:

> ❓ *Should I invest in Growth stocks or Value stocks?*

Some years, **Growth (IWP)** dominates. Other years, **Value (IWS)** takes the lead. The problem — no one knows **when to switch**.

Instead of guessing, this project asks:

> 💡 *Can machine learning learn this switching pattern automatically?*

---

## 🎯 Problem Statement

Traditional static strategies fail because markets **rotate between styles** — one ETF underperforms depending on the economic cycle.

| Strategy | Problem |
|---|---|
| Buy & Hold Growth (IWP) | Underperforms during value cycles |
| Buy & Hold Value (IWS) | Underperforms during growth cycles |
| **ML Rotation (this project)** | ✅ Adapts dynamically |

**Core Goal:** Build a system that dynamically selects the better-performing ETF at the right time.

---

## 🧠 Key Insight: Reframing the Problem

Instead of predicting:
> ❌ "Will the market go up or down?"

This project predicts:
> ✅ "Which ETF will outperform the other?"

This small shift simplifies the problem significantly and improves model performance.

---

## ⚙️ Methodology

### 1️⃣ Data Collection

- **Source:** Yahoo Finance via `yfinance`
- **ETFs:**
  - `IWP` → iShares Russell Mid-Cap Growth ETF
  - `IWS` → iShares Russell Mid-Cap Value ETF
- **Time Period:** 2002 – Present

---

### 2️⃣ Feature Engineering

Features engineered to capture key market behaviors:

| Feature | Description |
|---|---|
| **Momentum** | Recent price movement direction and strength |
| **Volatility** | Rolling standard deviation of returns (market risk) |
| **Trend** | Short-term vs. long-term moving average crossovers |
| **Relative Strength (RS)** ⭐ | `IWP_Close / IWS_Close` — the most important signal |

The **Relative Strength** feature was the biggest breakthrough:

```python
RS = IWP_Close / IWS_Close
```

> This directly captures which ETF is stronger *right now*, rather than trying to predict absolute returns.

---

### 3️⃣ Target Variable Design

```python
Target = 1  # IWP (Growth) outperforms next period
Target = 0  # IWS (Value) outperforms next period
```

The model learns **relative performance**, not absolute price direction.

---

### 4️⃣ Model: XGBoost Classifier

**Why XGBoost?**
- Handles noisy financial time-series data well
- Captures non-linear relationships between features
- Robust performance on tabular data without heavy tuning

---

## 🔄 Strategy Logic

**Step 1 — Predict Probability**

XGBoost outputs the probability that IWP will outperform IWS.

**Step 2 — Apply Confidence Thresholds**

```python
if prob > 0.63:
    signal = 1   # Invest in IWP (Growth)
elif prob < 0.37:
    signal = 0   # Invest in IWS (Value)
else:
    signal = 0.5 # HOLD — avoid low-confidence trades
```

The thresholds filter out uncertain signals and reduce unnecessary trading.

**Step 3 — Compute Strategy Return**

```python
Strategy_Return = (
    signal * IWP_Return +
    (1 - signal) * IWS_Return
)
```

Capital rotates dynamically based on the model's confidence.

---

## 📈 Results

### 💰 Portfolio Performance (Starting Value: $1.00)

| Strategy | Final Value |
|---|---|
| 🥇 ML Rotation Strategy | **$1.52** |
| 🥈 IWS (Value — Buy & Hold) | $1.37 |
| 🥉 IWP (Growth — Buy & Hold) | $1.21 |

### 📊 Risk Metrics

| Metric | Value |
|---|---|
| Sharpe Ratio | ~0.56 |
| Max Drawdown | ~-23.5% |
| Total Trades | 40 |

### 🔍 Interpretation

- The ML strategy **consistently outperforms both ETFs** over the full period
- It **adapts to changing market conditions** rather than getting stuck in one style
- Low trade count (40) means **realistic execution with reduced friction**
- The moderate Sharpe Ratio reflects a stable, non-overfitted model

---

## 🏁 Conclusion

> 📌 Markets rotate. Static strategies fail. Adaptive strategies win.

This project demonstrates that machine learning can identify *when* to rotate between investment styles — and that dynamic allocation can outperform traditional buy-and-hold approaches without excessive trading.

---

## 🚀 Future Improvements

- [ ] Add macroeconomic indicators (yield curve, CPI, Fed rate signals)
- [ ] Experiment with LSTM / deep learning models for sequential patterns
- [ ] Incorporate transaction costs and slippage into backtesting
- [ ] Deploy as a live Streamlit dashboard with real-time signals
- [ ] Extend to additional ETF pairs (e.g., small-cap, sector rotation)

---

**Requirements:**
```
yfinance
xgboost
pandas
numpy
scikit-learn
matplotlib
```

---

## 👩‍💻 Author

**Pooja Sukhdeve**
Master's in Computer Science — Boston University

