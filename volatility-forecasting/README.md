# 📉 Volatility Forecasting (GARCH vs Machine Learning)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Econometrics](https://img.shields.io/badge/Model-GARCH-orange)
![Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A quantitative finance project that forecasts **future market volatility** using both **classical econometric models** and **machine learning models**, and compares their performance.

The project predicts **future 5-day realized volatility of the S&P 500** using historical market behavior and engineered financial indicators.

Models compared in this project:

- Baseline rolling volatility
- **GARCH(1,1)** (traditional volatility model)
- **Random Forest** (machine learning approach)

---

# 🧠 Motivation

Volatility forecasting plays a crucial role in quantitative finance and risk management.

Applications include:

- Portfolio risk management
- Options pricing
- Volatility trading strategies
- Position sizing
- Risk forecasting
- Stress testing

Unlike price prediction, volatility modeling focuses on **risk dynamics**, which often exhibit stronger statistical structure such as **volatility clustering**.

This project explores whether **machine learning models can outperform classical econometric volatility models**.

---

# 📊 Models Compared

## 1️⃣ Baseline Model

The baseline predicts future volatility using **20-day rolling realized volatility**.

Volatility persistence makes this a strong benchmark.

---

## 2️⃣ GARCH(1,1)

The **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** model captures:

- volatility clustering  
- time-varying variance  
- persistence of shocks  

GARCH remains one of the most widely used volatility models in quantitative finance.

---

## 3️⃣ Random Forest

A **Random Forest Regressor** is used as a machine learning model for volatility forecasting.

The model learns nonlinear relationships between market indicators and future volatility.

---

# 📈 Target Variable

The model predicts:

```
Future 5-Day Realized Volatility
```

Computed as:

```
rolling_std(returns, 5).shift(-5)
```

This ensures:

- features use **past information**
- the target represents **future volatility**

which prevents **look-ahead bias**.

---

# 📊 Example Output

```
========================================================================
                  VOLATILITY FORECASTING PIPELINE
========================================================================

[ MODEL PERFORMANCE ]

            Model     RMSE      MAE
       GARCH(1,1) 0.006147 0.003899
Baseline (Vol 20) 0.006571 0.003937
    Random Forest 0.007159 0.003631
```

---

# 🔎 Key Findings

- **GARCH(1,1)** achieved the best **RMSE**, indicating the strongest overall volatility forecasting performance.
- The **rolling volatility baseline** remained competitive, highlighting the persistence of volatility in financial markets.
- **Random Forest** achieved the lowest **MAE**, but showed higher RMSE, suggesting sensitivity to larger volatility shocks.

This demonstrates that **classical econometric models remain highly competitive with machine learning models in financial volatility forecasting**.

---

# 📁 Project Structure

```
volatility-forecasting/
│
├── src/
│   ├── __init__.py
│   └── volatility_forecasting_pipeline.py
│
├── data/
│   ├── raw/
│   │   └── sp500_prices.csv
│   │
│   └── processed/
│       └── volatility_features.csv
│
├── notebooks/
│   └── volatility_exploration.ipynb
│
├── results/
│   ├── model_metrics.csv
│   ├── volatility_predictions.csv
│   ├── forecast_comparison.png
│   └── summary.json
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/<username>/quant-ml-lab.git
cd quant-ml-lab/volatility-forecasting
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Running the Pipeline

Run the forecasting pipeline:

```
python src/volatility_forecasting_pipeline.py
```

The script will:

1. Download S&P 500 historical data  
2. Generate volatility features  
3. Train forecasting models  
4. Evaluate model performance  
5. Save results and visualizations  

---

# 📊 Generated Outputs

### Model Metrics

```
results/model_metrics.csv
```

Contains RMSE and MAE for each model.

---

### Forecast Data

```
results/volatility_predictions.csv
```

Includes:

- actual volatility  
- baseline predictions  
- GARCH predictions  
- Random Forest predictions  

---

### Forecast Visualization

```
results/forecast_comparison.png
```

Displays predicted vs actual volatility over time.

---

# 🔬 Feature Engineering

The model uses financial features such as:

| Feature | Description |
|--------|-------------|
| log_return | Log price return |
| abs_return | Absolute return magnitude |
| vol_5 | 5-day rolling volatility |
| vol_10 | 10-day rolling volatility |
| vol_20 | 20-day rolling volatility |
| mean_return_5 | Rolling average return |
| momentum_5 | 5-day price momentum |
| momentum_10 | 10-day price momentum |

These features capture:

- volatility clustering  
- magnitude of price shocks  
- short-term trend behavior  

---

# 🔬 Potential Extensions

Possible future improvements:

- EGARCH / GJR-GARCH models
- Gradient boosting models (XGBoost / LightGBM)
- volatility regime detection
- multivariate volatility models
- options implied volatility integration
- crypto volatility forecasting

---

# ⚠️ Disclaimer

This project is intended **for research and educational purposes only**.

It should **not be considered financial advice**.

---

# 👨‍💻 Author

**Abhi Patidar**