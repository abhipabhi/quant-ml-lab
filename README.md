# 🧠 Quant ML Lab

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Finance-green)
![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-purple)
![Research](https://img.shields.io/badge/Type-Research%20Lab-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Quant ML Lab** is a long-term **quantitative finance research repository** focused on studying financial markets using:

- machine learning  
- econometrics  
- statistical modeling  
- trading strategy research  

This repository functions as a **personal quantitative research laboratory**, where models, strategies, and financial analytics tools are developed and tested on real market data.

The goal is to build **practical research systems**, not just isolated scripts.

---

# 🔬 Research Areas

The repository is organized around core areas of quantitative finance.

### Market Regime Modeling

Detect global market regimes such as **bull, bear, high-volatility, and transition states**.

Techniques explored include:

- Hidden Markov Models  
- cross-market indicators  
- volatility signals  
- trend indicators  
- regime transition modeling  

Project:

```
research/market_regime_detection
```

---

### Volatility Modeling

Forecast future market volatility using both **econometric models and machine learning approaches**.

Applications include:

- risk forecasting  
- derivatives pricing  
- volatility trading strategies  

Project:

```
research/volatility_forecasting
```

Models explored:

- GARCH models  
- rolling volatility models  
- machine learning regressors  

---

### Statistical Arbitrage

Study **market-neutral trading strategies** that exploit temporary pricing inefficiencies.

Project:

```
research/pairs_trading_cointegration
```

Topics explored:

- cointegration testing  
- spread modeling  
- mean-reversion signals  
- pairs trading backtesting  

---

# 📊 Research Projects

| Project | Area | Description |
|------|------|------|
| Market Regime Detection | Regime Modeling | Detect global financial regimes using Hidden Markov Models and multi-market indicators |
| Volatility Forecasting | Risk Modeling | Forecast future market volatility using GARCH and machine learning models |
| Pairs Trading Cointegration | Statistical Arbitrage | Build and backtest a mean-reversion pairs trading strategy using cointegration |

More projects will be added as research expands.

---

# 📂 Repository Structure

The repository is designed as a **scalable research lab** where new financial experiments and models can be added over time. Each project under `research/` is self-contained, with its own `src/`, `data/`, `notebooks/`, `results/`, and README.

```
quant-ml-lab/
│
├── research/                          # Self-contained research projects
│   ├── market_regime_detection/       # HMM regime classification
│   ├── volatility_forecasting/        # GARCH vs ML volatility forecasting
│   └── pairs_trading_cointegration/   # Cointegration-based pairs trading
│
├── apps/                              # Applications built on research outputs
│   └── quant_research_terminal/       # Streamlit dashboard
│
├── requirements.txt                   # Superset of all project dependencies
├── LICENSE
└── README.md
```

Each research project follows the same internal layout:

```
<project>/
├── src/                 # Pipeline / model code
├── data/                # raw/ and processed/ datasets
├── notebooks/           # Exploratory analysis
├── results/             # Saved metrics, charts, and JSON summaries
├── README.md
└── requirements.txt
```

---

# 🚀 Getting Started

Clone the repository and install the shared dependencies (a superset covering every project):

```bash
git clone https://github.com/abhipabhi/quant-ml-lab.git
cd quant-ml-lab
pip install -r requirements.txt
```

Each project runs from the repository root as a standalone pipeline. The pipelines
download market data via `yfinance`, so an internet connection is required, and
they write their outputs into that project's `results/` folder.

```bash
# Market regime detection (HMM)
python research/market_regime_detection/src/current_world_market_regime.py

# Volatility forecasting (Baseline vs GARCH vs Random Forest)
python research/volatility_forecasting/src/volatility_forecasting_pipeline.py

# Pairs trading (cointegration test + backtest)
python research/pairs_trading_cointegration/src/pairs_trading_pipeline.py
```

Launch the unified dashboard once the pipelines have produced results:

```bash
streamlit run apps/quant_research_terminal/app.py
```

> A shared `src/` package for reusable loaders, indicators, and backtesting
> utilities is planned as common code is factored out of the individual projects.

---

# 🗺 Research Roadmap

This repository will continue expanding as new quantitative finance experiments are developed.

### Market Modeling

- regime detection  
- volatility forecasting  
- macro regime classification  

### Trading Strategies

- statistical arbitrage  
- momentum strategies  
- mean reversion strategies  
- multi-asset portfolio allocation  

### Financial Machine Learning

- feature engineering for financial data  
- regime-aware ML models  
- reinforcement learning for trading  

### Risk & Portfolio Modeling

- portfolio optimization  
- factor models  
- drawdown risk modeling  

### Market Microstructure

- order book modeling  
- liquidity analysis  
- execution strategies  

---

# 📚 Research Principles

Projects in this repository follow a research-oriented methodology:

- use real market data whenever possible  
- compare models against strong statistical baselines  
- avoid look-ahead bias and data leakage  
- evaluate strategies using proper backtesting metrics  
- focus on reproducible experiments  

---

# ⚙️ Technologies Used

Core tools used across projects include:

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Statsmodels  
- Financial time-series analysis  
- Data visualization  

---

# 🎯 Purpose

This repository serves as:

- a **quantitative research lab**
- a **machine learning portfolio**
- a **collection of financial data science experiments**
- a **foundation for building trading and analytics systems**

---

# ⚠️ Disclaimer

This repository is intended **for research and educational purposes only**.

It should **not be considered financial advice**.

---

# 👨‍💻 Author

**Abhi Patidar**
