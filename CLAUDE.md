# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A monorepo of independent quantitative-finance research projects. There is no shared
build system, package, or test suite — each project is a set of standalone scripts that
pull live market data, run a model, and write artifacts to disk. A Streamlit app reads
those artifacts back as a dashboard.

## Layout

- `research/<project>/` — three self-contained projects, each with `src/` (pipeline),
  `data/` (generators + raw/processed CSVs), `notebooks/`, `results/`, and its own README:
  - `market_regime_detection` — Gaussian HMM over multi-market features (`hmmlearn`)
  - `volatility_forecasting` — Baseline vs GARCH(1,1) (`arch`) vs RandomForest, compared by RMSE/MAE
  - `pairs_trading_cointegration` — OLS hedge ratio → Engle-Granger cointegration → z-score backtest
- `apps/quant_research_terminal/app.py` — Streamlit dashboard (read-only over `results/`)
- `requirements.txt` (root) — superset of every project's deps; each project also has its own subset

## Running pipelines — output paths are CWD-relative (important)

Every pipeline and generator writes to **relative** paths (`Path("data/raw")`, `Path("results")`),
resolved against the current working directory, **not** the script location. Run each pipeline
from **inside its project directory** so outputs land in that project's `data/` and `results/`
(which is where the committed artifacts live and where the Streamlit app reads them):

```bash
cd research/market_regime_detection      && python src/current_world_market_regime.py
cd research/volatility_forecasting       && python src/volatility_forecasting_pipeline.py
cd research/pairs_trading_cointegration  && python src/pairs_trading_pipeline.py
```

Running from the repo root instead scatters `data/` and `results/` into the root — the wrong
place. There is no `argparse`; change tickers/period/thresholds via the pipeline class
constructor (e.g. `PairsTradingPipeline(ticker_y=..., ticker_x=...)`) in `main()`.

## Pipeline vs. generator

Each `src/*.py` pipeline is **self-contained**: it re-fetches its own data via `yfinance`,
builds features, runs the model, and saves both datasets and results. You do **not** need to
run the `data/generate_*.py` script first — those are lighter standalone dataset builders for
exploration. Note the pairs generator builds a naive `KO − PEP` spread, while the pipeline uses
the OLS hedge-ratio spread `y − β·x`; they are not identical.

## Streamlit app

`app.py` anchors to the repo via `PROJECT_ROOT = APP_DIR.parents[1]` and only **reads** each
project's `results/` (JSON/CSV/PNG). It never runs a pipeline — run the pipelines first or the
dashboard shows "Missing Outputs". Launch from anywhere:

```bash
streamlit run apps/quant_research_terminal/app.py
```

## Shared pipeline conventions

The three `src/` pipelines follow the same shape — understanding one transfers to the others:
class configured in `__init__` → `fetch_*` (yfinance, `auto_adjust=True`, handles both flat and
MultiIndex `Close` columns) → `build_*` features → model → `evaluate`/`compute_metrics` → `save_*`
(a JSON summary + CSV + PNG chart), returning a `@dataclass` result. All suppress warnings with
`warnings.filterwarnings("ignore")`.

Time-series discipline is deliberate and should be preserved: splits are chronological (no
shuffling), features use only past data, targets are shifted into the future (e.g. volatility's
`target_vol_5 = rolling(5).std().shift(-5)`), and the GARCH forecast refits on an expanding
window per test step. Don't introduce shuffled splits or let target windows leak into features.

## No tests / lint / CI

There is no test suite, linter, or CI. To sanity-check a change without the network round-trip,
compile it: `python -m py_compile research/<project>/src/*.py`. Full verification requires running
a pipeline, which needs internet access (live `yfinance` downloads); results drift as market data
updates, so exact metric reproduction is not expected.

## Known caveat

The default KO/PEP pair is **not** cointegrated (Engle-Granger p ≈ 0.95). The pairs backtest is a
demonstration of the methodology, not evidence of an edge — don't present its metrics as a real
strategy result.
