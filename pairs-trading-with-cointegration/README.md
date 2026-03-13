# 🔗 Pairs Trading with Cointegration

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Statistics](https://img.shields.io/badge/Method-Cointegration-green)
![Trading](https://img.shields.io/badge/Strategy-Pairs%20Trading-orange)
![Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A quantitative finance project that builds a **pairs trading dataset** using two historically related assets and prepares the foundation for a **statistical arbitrage strategy**.

The objective is to identify whether two assets move together over time, construct their spread, and generate processed data for testing **mean-reversion trading signals**.

This project serves as the dataset and research foundation for a full pairs trading pipeline.

---

# 🧠 Motivation

Pairs trading is a classic **statistical arbitrage** strategy used in quantitative finance.

The idea is simple:

- choose two related assets
- test whether they are statistically linked
- measure deviations between them
- trade when the spread moves abnormally far from its historical relationship

This project prepares the raw and processed dataset required for:

- cointegration testing
- spread modeling
- z-score signal generation
- backtesting
- performance analysis

---

# 📊 Assets Used

The first version of this project uses:

- **KO** — Coca-Cola  
- **PEP** — PepsiCo  

This pair is a common example because both companies operate in a similar industry and often exhibit related price behavior.

---

# 📈 Dataset Construction

The dataset pipeline performs the following steps:

1. Download adjusted closing prices for both assets  
2. Align the price series by date  
3. Compute daily log returns  
4. Estimate the price spread  
5. Compute rolling mean and rolling standard deviation of the spread  
6. Compute spread z-score  
7. Save raw and processed datasets  

The resulting processed dataset can later be used to generate trading signals and build a pairs trading strategy.

---

# 📁 Project Structure

```
pairs-trading-cointegration/
│
├── src/
│   ├── __init__.py
│   └── pairs_trading_pipeline.py
│
├── data/
│   ├── raw/
│   │   └── ko_pep_prices.csv
│   │
│   └── processed/
│       └── ko_pep_spread_dataset.csv
│
├── notebooks/
│   └── pairs_trading_exploration.ipynb
│
├── results/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/<username>/quant-ml-lab.git
cd quant-ml-lab/pairs-trading-cointegration
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Running the Dataset Generator

Run the dataset generation script:

```
python generate_pairs_dataset.py
```

The script will:

1. Download historical prices for KO and PEP  
2. Save the raw price data  
3. Build the spread dataset  
4. Save the processed dataset  

---

# 📊 Generated Outputs

### Raw Price Data

```
data/raw/ko_pep_prices.csv
```

Contains aligned daily close prices for both stocks.

---

### Processed Spread Dataset

```
data/processed/ko_pep_spread_dataset.csv
```

Contains:

- asset prices  
- log returns  
- spread  
- rolling spread statistics  
- z-score of the spread  

---

# 🔬 Features in the Processed Dataset

| Feature | Description |
|--------|-------------|
| KO_close | Coca-Cola closing price |
| PEP_close | PepsiCo closing price |
| ko_return | KO log return |
| pep_return | PEP log return |
| spread | Price spread between the two assets |
| spread_mean_20 | 20-day rolling mean of spread |
| spread_std_20 | 20-day rolling standard deviation of spread |
| spread_zscore | Standardized spread deviation |

These features are the basis for later:

- entry/exit signals  
- long/short spread trades  
- mean-reversion backtesting  

---

# 🔬 Potential Extensions

Possible next steps include:

- Engle-Granger cointegration test  
- hedge ratio estimation with OLS  
- z-score trading signals  
- backtesting engine  
- Sharpe ratio and drawdown evaluation  
- rolling hedge ratio estimation  

---

# ⚠️ Disclaimer

This project is intended **for research and educational purposes only**.

It should **not be considered financial advice**.

---

# 👨‍💻 Author

**Abhi Patidar**