# ETF Momentum, KAMA and EPO Backtest

A reproducible research prototype for monthly ETF allocation. It ranks ETFs by regression momentum, optionally filters candidates with Kaufman's Adaptive Moving Average (KAMA), and allocates across up to three selected ETFs using an anchored EPO formula. The backtest includes transaction costs, cash interest, financing costs, portfolio accounting, and attribution reports.

This repository contains research code, not a live trading system. Historical results are sensitive to the chosen universe, parameters, data quality, and execution assumptions.

## How the strategy works

1. **Rank:** Fit a line to log adjusted closing prices over each momentum window. The score is `expm1(250 × slope) × R²`. The default window contains 34 closing prices. With multiple windows, each window is ranked separately and the average rank determines the order; a strict majority of scores must be positive.
2. **Filter:** For `Momentum_kama`, require the fast KAMA to exceed the slow KAMA and to have risen over the past five trading days. Set `USE_KAMA_FILTER = False` to skip this filter. `Momentum_EQ` and `Momentum_EPO_34D` never use KAMA.
3. **Select:** Take up to `TOP_N = 3` eligible ETFs. If none qualifies, allocate 100% to BIL. With KAMA enabled, `KAMA_FILTER_DEFENSIVE = False` removes BIL from normal selection but keeps this fallback.
4. **Allocate:** `Momentum_kama` applies inverse-variance anchoring, covariance shrinkage, and an endogenous signal scale to determine nonnegative EPO weights. If its inputs are insufficient or unstable, it records the reason and uses equal weights. The EPO lookback defaults to 1,300 closing prices.
5. **Trade:** Rebalance at the first available opening price of each month, using only information through the previous trading day's close. The three momentum strategies target 100% invested exposure and do not use account leverage.

| Output name | Selection | Allocation | KAMA |
| --- | --- | --- | --- |
| `Momentum_kama` | Configured regression momentum windows | Anchored EPO | Configurable; on by default |
| `Momentum_EQ` | Configured regression momentum windows | Equal weights | No |
| `Momentum_EPO_34D` | Fixed 34-price window | Anchored EPO | No |
| `EqualWeight` | Entire configured universe | Monthly equal weights | No |
| `LeveragedEqualWeight` | Non-BIL ETFs only | Optional leveraged equal weights | No |

The optional leveraged benchmark is off by default. If enabled, `MAX_LEVERAGE` applies **only** to `LeveragedEqualWeight`; BIL has zero weight in that benchmark. It does not change leverage in the momentum strategies. This benchmark differs from `EqualWeight` in both exposure and asset universe.

## Quick start

Python 3.10+ is recommended. Internet access is needed for the initial Yahoo Finance download through `yfinance`; subsequent runs use cached CSV files unless `REFRESH_DATA = True`.

```bash
git clone https://github.com/YOUR_USERNAME/etf-momentum-kama-epo.git
cd etf-momentum-kama-epo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python QF5210_KAMA.py
```

For a short demonstration of the EPO calculation without downloading market data:

```bash
python QF5210_KAMA.py --epo-example
```

Edit the parameter block near the top of `QF5210_KAMA.py` before a run. The default `DATA_END = "2026-10-05"` is **exclusive**; update it to extend the sample. Set `SHOW_PLOTS = False` when running without a desktop display. No TA-Lib installation is required.

## Key settings

| Setting | Default | Effect |
| --- | --- | --- |
| `USE_EXPANDED` | `True` | Use the expanded ETF universe; `False` selects the core universe. |
| `BACKTEST_START` | `2016-01-01` | First eligible backtest date, after the signal lookback. |
| `MOMENTUM_WINDOWS` | `[34]` | Number of closing prices per regression window. |
| `USE_KAMA_FILTER` | `True` | Apply KAMA eligibility to `Momentum_kama`. |
| `KAMA_FAST`, `KAMA_SLOW`, `KAMA_DIFF_DAYS` | `(20, 5, 30)`, `(60, 15, 60)`, `5` | KAMA efficiency, smoothing, and rising-trend settings. |
| `TOP_N` | `3` | Maximum number of eligible ETFs held. |
| `EPO_PRICE_DAYS`, `EPO_SHRINKAGE` | `1300`, `0.20` | EPO historical price window and covariance shrinkage. |
| `COST_BPS` | `10` | Estimated cost per dollar bought **or** sold. |
| `CASH_RATE`, `BORROW_SPREAD` | `0.02`, `0.01` | Constant annual positive-cash rate and extra annual borrowing spread. |
| `RUN_LEVERAGED_EW`, `MAX_LEVERAGE` | `False`, `1.25` | Optional non-BIL leveraged equal-weight benchmark only. |
| `EXPORT_DETAILS` | `True` | Save daily ledgers, holdings, scores, decisions, and diagnostics. |

## Outputs

The script writes a parameter-specific `results_*/` directory. Start with `REPORT_CN.md` for a Chinese summary and the exact settings used. Important files include:

- `performance.csv`, `nav.csv`, `yearly_returns.csv`: metrics, net asset values, and calendar-year returns.
- `cost_summary.csv`: net-versus-zero-cost results from separate backtest runs.
- `asset_contribution_full.csv`, `contribution_by_period.csv`, `holding_summary.csv`: realized attribution and exposures when `RUN_ATTRIBUTION = True`.
- `backtest.png`, `momentum_comparison.png`, `contribution_comparison.png`: plots. The comparison changes both KAMA screening and EPO weighting, so it does not isolate either effect.
- `details/`: optional daily ledgers and signal decisions. `data/` contains downloaded raw prices.

Generated data and results are excluded from Git by `.gitignore`. To publish a specific chart or performance table, review and add it separately with its data end date and configuration.

## Research limitations

- Adjusted opening prices are approximated from raw opens and the adjusted-close factor. Vendor revisions and missing data can change results.
- All ETF tickers are traded as listed instruments; international equity ETFs can reflect underlying markets that closed at different times. The previous-close signal rule does not by itself resolve every cross-market timing issue.
- The ETF universe is fixed in hindsight. The 34-price window and other settings were explored on historical data; the same history is not a fresh out-of-sample test.
- Financing and cash rates are constant assumptions. Explicit costs are charged per traded dollar, with no separate market-impact or capacity model.
- Inverse ETFs in the expanded universe are held as long positions in those ETFs. Their daily-reset behavior can make longer holding-period returns differ from the inverse of an index return.

Use a genuinely untouched future period and independently verified market data before interpreting any backtest as evidence of deployable performance.
