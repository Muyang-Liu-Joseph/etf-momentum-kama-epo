"""QF5210 ETF 回测：回归动量、可选 KAMA 筛选与 EPO 配权。

安装：python -m pip install numpy pandas matplotlib yfinance
运行：python QF5210_KAMA.py

先读运行后输出目录中的 REPORT_CN.md；详细输出由 EXPORT_DETAILS 控制。
不同窗口和 KAMA 参数保存在不同结果目录。
逐资产净贡献 = 实际持仓盈亏 - 分摊到该资产的实际交易费用。
各期贡献以期初净资产为分母，合计等于该期累计收益，不是年化收益。
零费用版本单独重跑，用于比较真正的费用情景差异。

默认 MOMENTUM_WINDOWS=[34]，保留原回归动量；改成 [63,126,252] 即启用多周期。
单窗口按原回归分数排序；多窗口分别排名再取平均名次，越小越优。
多周期默认至少过半窗口的分数为正；合格候选最多持有 3 只，在每月首个交易日开盘调仓。
Momentum_EQ 不使用 KAMA；Momentum_kama 使用 EPO 配权，可通过开关启用 KAMA 筛选。
这两条策略目标 ETF 权重均为 100%，没有账户杠杆或目标波动率控制。
Momentum_EPO_34D 使用同一资产池的 34 日信号和 EPO，但不使用 KAMA。
可选杠杆等权基准只买非 BIL 资产；BIL 不与负现金/融资同时持有。
EPO 使用逆方差锚定、内生信号缩放及协方差收缩；不是约束优化器。
默认买卖各 10bp；复权开盘是总收益近似；利率为常数情景假设。
历史回测不自动构成未经选择的样本外证据，最终需预留未调参区间。
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 1. Parameters ====================
CORE = ["SPY", "EEM", "EWS", "GLD", "DBC", "BIL", "DBA"]
EXTRA = [
    "QQQ", "IWM", "XLK", "XLF", "SMH", "USO", 
    "EWY",   # 韩国
    "EWJ",   # 日本
    "EWH",   # 香港 
    "EWU",   # 英国
    "SH", "PSQ", "RWM",  # 分别跟踪标普500、纳指100、罗素2000单日反向收益的 ETF
]
USE_EXPANDED = True          # True: 扩展池；False: 基础池
TICKERS = CORE + EXTRA if USE_EXPANDED else CORE

DATA_START = "2008-01-01"
DATA_END = "2026-10-05"        # Exclusive: no partial 2026-10-05 data.
BACKTEST_START = "2016-01-01"
COST_BPS = 10.0              # Fee/slippage assumption per dollar bought OR sold.
CASH_RATE = 0.02             # Constant scenario assumption, not historical USD rates.
BORROW_SPREAD = 0.01
MAX_LEVERAGE = 1.25          # 只用于可选的非 BIL 资产杠杆等权基准；两条动量策略上限固定为 1。
RUN_LEVERAGED_EW = False     # 默认不运行杠杆基准；打开后 BIL 权重为零。
REFRESH_DATA = False
SHOW_PLOTS = True
RUN_COST_COUNTERFACTUAL = True  # KAMA+EPO、动量等权和无 KAMA 的 34 日 EPO 分别重跑零费用情景。
RUN_ATTRIBUTION = True
EXPORT_DETAILS = True      # True 保存逐日账本、各窗口分数/排名和 EPO 输入；False 只留摘要。

# 按持有区间的结束日期选取阶段；end 为不包含边界。
ATTRIBUTION_PERIODS = {
    "2018_2020": ("2018-01-01", "2021-01-01"),
    "2022_2023": ("2022-01-01", "2024-01-01"),
    "2025_2026": ("2025-01-01", "2027-01-01"),
}

# ETF 分类和敞口代理，不穿透成分股；SPY 中也有科技股。
ASSET_GROUPS = {
    "Tech_related_ETFs": ["QQQ", "XLK", "SMH"],
    "Other_US_equity_ETFs": ["SPY", "IWM", "XLF"],
    "International_equity": ["EEM", "EWS", "EWY", "EWJ", "EWH", "EWU"],
    "Gold": ["GLD"], "Broad_commodities": ["DBC"],
    "Oil": ["USO"], "Agriculture": ["DBA"], "Cash_ETF": ["BIL"],
}

STRATEGY_PLOT_STYLES = {
    "Momentum_kama": {"color": "#0072B2", "linewidth": 3.4,
                     "linestyle": "-", "alpha": 1.0, "zorder": 11},
    "Momentum_EQ": {"color": "#E69F00", "linewidth": 3.0,
                    "linestyle": "--", "alpha": 1.0, "zorder": 10},
}

MOMENTUM_WINDOWS = [34]     # 主版本
USE_KAMA_FILTER = True
# KAMA 参数统一在这里修改：(效率比周期, 快速平滑周期, 慢速平滑周期)
# 自定义递推支持全部三个参数；首个有效值以上一日收盘价为初始值。
KAMA_FAST = (20, 5, 30)
KAMA_SLOW = (60, 15, 60)
KAMA_DIFF_DAYS = 5  # 快线相对 5 个交易日前必须上升
KAMA_FILTER_DEFENSIVE = True  # False: KAMA 开启时 BIL 不参加正常选取；无合格资产仍回退 BIL。
MIN_POSITIVE_WINDOWS = None  # None=严格过半；三个窗口默认至少两个为正。也可指定整数。
MOMENTUM_ANNUALIZATION = 250
TOP_N = 3
EPO_PRICE_DAYS = 1300        # 1300 个收盘价格产生 1299 个日收益。
EPO_SHRINKAGE = 0.20
DEFENSIVE_ETF = "BIL"        # 无合格候选时持有 BIL；上述开关决定 KAMA 开启时能否正常入选。

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAW_DIR = ROOT / "data" / f"raw_{DATA_START}_{DATA_END}"
WINDOW_TAG = "_".join(map(str, MOMENTUM_WINDOWS))
KAMA_TAG = (f"_KAMA_F{'_'.join(map(str, KAMA_FAST))}"
            f"_S{'_'.join(map(str, KAMA_SLOW))}_D{KAMA_DIFF_DAYS}"
            f"_BIL{int(KAMA_FILTER_DEFENSIVE)}" if USE_KAMA_FILTER else "")
OUT_DIR = ROOT / (("results_expanded" if USE_EXPANDED else "results_core")
                  + f"_v4_no_long_bonds_mom_{WINDOW_TAG}" + KAMA_TAG)
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# ==================== 2. Download and clean ====================
def load_data():
    import yfinance as yf
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, audit = {}, []
    for ticker in TICKERS:
        path = RAW_DIR / f"{ticker}.csv"
        if path.exists() and not REFRESH_DATA:
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        else:
            print(f"Downloading {ticker} ...")
            df = yf.download(ticker, start=DATA_START, end=DATA_END,
                             auto_adjust=False, multi_level_index=False,
                             actions=True, keepna=True, threads=False,
                             progress=False, timeout=30)
            if df is None or df.empty:
                raise RuntimeError(f"No data for {ticker}; please retry.")
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=-1)
            df.to_csv(path, index_label="Date")
        if not set(FIELDS).issubset(df.columns):
            raise ValueError(f"{ticker}: missing fields {FIELDS}")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        if df.index.has_duplicates:
            raise ValueError(f"{ticker}: duplicated dates")
        df = df.sort_index().loc[DATA_START:]
        df = df.loc[df.index < pd.Timestamp(DATA_END)].copy()
        df[FIELDS] = df[FIELDS].apply(pd.to_numeric, errors="coerce")
        p = df[FIELDS[:5]]
        bad = (~np.isfinite(df[FIELDS]).all(axis=1) | p.le(0).any(axis=1)
               | df.Volume.lt(0)
               | df.High.lt(df[["Open", "Close", "Low"]].max(axis=1))
               | df.Low.gt(df[["Open", "Close", "High"]].min(axis=1)))
        df.loc[bad, FIELDS] = np.nan
        good = df[FIELDS].notna().all(axis=1)
        if not good.any():
            raise ValueError(f"{ticker}: no valid observations")
        audit.append({"ticker": ticker, "invalid_rows": int(bad.sum()),
                      "first": df.index[good][0], "last": df.index[good][-1]})
        frames[ticker] = df
    audit = pd.DataFrame(audit)
    if EXPORT_DETAILS:
        (OUT_DIR / "details").mkdir(parents=True, exist_ok=True)
        audit.to_csv(OUT_DIR / "details" / "data_audit.csv", index=False)
    first, last = audit["first"].max(), audit["last"].min()
    dates = frames["SPY"].index
    dates = dates[(dates >= first) & (dates <= last)]
    if len(dates) < 2:
        raise ValueError("Insufficient common sample")
    missing = []
    for ticker in TICKERS:
        frames[ticker] = frames[ticker].reindex(dates)
        bad = frames[ticker][FIELDS].isna().any(axis=1)
        missing += [{"ticker": ticker, "date": d} for d in dates[bad]]
    if missing:
        pd.DataFrame(missing).to_csv(OUT_DIR / "missing_data.csv", index=False)
        raise ValueError(f"Internal missing data: inspect {OUT_DIR / 'missing_data.csv'}; do not fill prices.")
    close = pd.DataFrame({t: frames[t]["Adj Close"] for t in TICKERS})
    adj_open = pd.DataFrame({t: frames[t].Open * frames[t]["Adj Close"] / frames[t].Close
                            for t in TICKERS})
    volume = pd.DataFrame({t: frames[t].Volume for t in TICKERS})
    if EXPORT_DETAILS:
        close.to_csv(OUT_DIR / "details" / "adjusted_close.csv", index_label="Date")
        adj_open.to_csv(OUT_DIR / "details" / "adjusted_open.csv", index_label="Date")
    print(f"Common sample: {dates[0].date()} to {dates[-1].date()}, {len(dates)} rows")
    kama_cache = precompute_kama(close) if USE_KAMA_FILTER else None
    return close, adj_open, volume, kama_cache


# ==================== 3. Regression momentum, KAMA and anchored EPO ====================
def regression_momentum(history, days):
    """原版单窗口公式；days 个收盘价格包含 days-1 个收益区间。"""
    rows = []
    for ticker in history.columns:
        p = history[ticker].tail(days).to_numpy(float)
        slope, r2, annual, score = np.nan, np.nan, np.nan, np.nan
        valid = (len(p) == days and np.isfinite(p).all()
                 and (p > 0).all())
        if valid:
            y = np.log(p)
            x = np.arange(len(y), dtype=float)
            xc, yc = x - x.mean(), y - y.mean()
            sst = yc @ yc
            if sst > 1e-20:  # 完全不变的价格无有效回归分数。
                slope = float(xc @ yc / (xc @ xc))
                residual = yc - slope * xc
                r2 = float(np.clip(1 - residual @ residual / sst, 0, 1))
                with np.errstate(over="ignore", invalid="ignore"):
                    annual = float(np.expm1(MOMENTUM_ANNUALIZATION * slope))
                score = annual * r2
                if not np.isfinite(score):
                    score = np.nan
        rows.append({"ticker": ticker, "slope": slope, "r_squared": r2,
                     "annualized_trend": annual, "score": score})
    # 稳定排序：相同分数时沿用 universe 的顺序，保证可复现。
    return (pd.DataFrame(rows).set_index("ticker")
            .sort_values("score", ascending=False, kind="stable", na_position="last"))


def momentum_settings(windows=None, min_positive=None):
    """窗口长度与正趋势门槛在一处配置；默认要求严格过半窗口为正。"""
    windows = list(MOMENTUM_WINDOWS if windows is None else windows)
    if (not windows or any(not isinstance(n, (int, np.integer)) or n < 3 for n in windows)
            or len(set(windows)) != len(windows)):
        raise ValueError("MOMENTUM_WINDOWS must contain distinct integers >= 3")
    configured = MIN_POSITIVE_WINDOWS if min_positive is None else min_positive
    required = len(windows)//2 + 1 if configured is None else configured
    if not isinstance(required, (int, np.integer)) or not 1 <= required <= len(windows):
        raise ValueError("MIN_POSITIVE_WINDOWS must be None or an integer in [1, number of windows]")
    return windows, int(required)


def momentum_scores(history, windows=None, min_positive=None):
    """各窗口按原回归分数排名，1 为最佳；多窗口取等权平均名次。

    单窗口保留原 score，正分数入选。多窗口 score=1/平均名次，仅用于排序；
    是否合格以 eligible 判断，不能对多窗口 score 使用 >0 代替正趋势筛选。
    所有指定窗口均须有有效数据；并列采用平均排名，最终并列沿用资产池顺序。
    """
    windows, required = momentum_settings(windows, min_positive)
    frames = {n: regression_momentum(history, n).reindex(history.columns) for n in windows}
    raw_scores = pd.DataFrame({n: frames[n].score for n in windows})
    valid = raw_scores.notna().all(axis=1)
    ranks = raw_scores.where(valid, np.nan, axis=0).rank(ascending=False, method="average")
    result = frames[windows[0]].copy() if len(windows) == 1 else pd.DataFrame(index=history.columns)
    for n in windows:
        for col in ["slope", "r_squared", "annualized_trend", "score"]:
            result[f"{col}_{n}"] = frames[n][col]
        result[f"rank_{n}"] = ranks[n]
    result["average_rank"] = ranks.mean(axis=1, skipna=False)
    result["positive_windows"] = raw_scores.gt(0).sum(axis=1)
    result["required_positive_windows"] = required
    result["valid"] = valid
    result["eligible"] = valid & result.positive_windows.ge(required)
    if len(windows) > 1:
        result["score"] = 1 / result.average_rank
    result.index.name = "ticker"
    return result.sort_values("score", ascending=False, kind="stable", na_position="last")


def epo_from_moments(signal, covariance, shrinkage=0.2):
    """逆方差锚定的 EPO，使用内生 gamma 调整信号尺度。

    s: 日收益均值；Sigma: 同一日收益样本的协方差（ddof=1）。
    V = diag(diag(Sigma)); a = normalized inverse variance.
    Sigma_tilde = (1-alpha)*Sigma + alpha*V.
    gamma = sqrt((a.T @ Sigma_tilde @ a)/(s.T @ solve(Sigma_tilde,s))).
    raw = solve(Sigma_tilde, (1-alpha)*gamma*s + alpha*V@a).
    非负截断，再归一化。lambda 在此分支中不使用。
    日/年单位都可，但收益与协方差必须采用同一时间单位。
    """
    s = np.asarray(signal, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if s.ndim != 1 or len(s) == 0 or cov.shape != (len(s), len(s)):
        raise ValueError("Invalid EPO input shape")
    if not 0 <= shrinkage <= 1:
        raise ValueError("EPO shrinkage must be in [0, 1]")
    if not np.isfinite(s).all() or not np.isfinite(cov).all():
        raise ValueError("Non-finite EPO input")
    variance = np.diag(cov)
    if (variance <= 1e-16).any():
        raise ValueError("Zero/near-zero variance")
    V = np.diag(variance)
    inv_variance = 1 / variance
    anchor = inv_variance / inv_variance.sum()
    tilde = (1 - shrinkage) * cov + shrinkage * V
    # 协方差矩阵不正定或病态时，由 epo_weights 记录原因并退回等权。
    np.linalg.cholesky(tilde)
    if np.linalg.cond(tilde) > 1e12:
        raise ValueError("Ill-conditioned shrunk covariance")
    numerator = float(anchor @ tilde @ anchor)
    if shrinkage == 1:
        gamma = 0.0  # full shrinkage 时 raw = anchor，与收益信号无关。
        raw = anchor.copy()
        adjusted_signal = V @ anchor
    else:
        denominator = float(s @ np.linalg.solve(tilde, s))
        if denominator <= 1e-20 or numerator <= 0:
            raise ValueError("Zero/near-zero EPO signal")
        gamma = float(np.sqrt(numerator / denominator))
        adjusted_signal = ((1 - shrinkage) * gamma * s
                           + shrinkage * (V @ anchor))
        raw = np.linalg.solve(tilde, adjusted_signal)
    positive = np.maximum(raw, 0)
    if not np.isfinite(raw).all() or positive.sum() <= 1e-12:
        raise ValueError("No positive EPO weight")
    weights = positive / positive.sum()
    details = {"signal": s, "covariance": cov, "variance": variance,
               "anchor": anchor, "shrunk_covariance": tilde,
               "gamma": gamma, "adjusted_signal": adjusted_signal,
               "raw_weights": raw, "weights": weights}
    return weights, details


def epo_weights(history, selected):
    """使用最后 EPO_PRICE_DAYS 个价格；无法配权时记录等权回退原因。"""
    n = len(selected)
    if n == 0:
        raise ValueError("EPO needs at least one selected ETF")
    if n == 1:
        return pd.Series([1.0], index=selected), {"status": "single_asset"}
    equal = pd.Series(1 / n, index=selected)
    prices = history[selected].tail(EPO_PRICE_DAYS)
    if len(prices) < EPO_PRICE_DAYS:
        return equal, {"status": "fallback_equal", "reason": f"insufficient_{EPO_PRICE_DAYS}_prices"}
    if not np.isfinite(prices.to_numpy()).all() or prices.le(0).any().any():
        return equal, {"status": "fallback_equal", "reason": "invalid_risk_prices"}
    returns = prices.pct_change(fill_method=None).iloc[1:]
    try:
        weights, details = epo_from_moments(
            returns.mean().to_numpy(), returns.cov().to_numpy(), EPO_SHRINKAGE)
    except (ValueError, np.linalg.LinAlgError) as error:
        return equal, {"status": "fallback_equal", "reason": str(error)}
    details.update(status="epo", risk_first_close=str(prices.index[0].date()),
                   risk_last_close=str(prices.index[-1].date()), observations=len(returns))
    return pd.Series(weights, index=selected), details


def validate_kama_parameters():
    """启动时检查参数，避免下载行情后才因参数错误失败。"""
    for name, params in (("KAMA_FAST", KAMA_FAST), ("KAMA_SLOW", KAMA_SLOW)):
        if (not isinstance(params, (tuple, list)) or len(params) != 3
                or any(not isinstance(v, (int, np.integer)) for v in params)):
            raise ValueError(f"{name} 必须是三个整数，例如 (20, 2, 30)，当前为 {params!r}")
        period, fast, slow = params
        if period < 2 or fast < 1 or fast >= slow:
            raise ValueError(f"{name} 参数不合法：需 timeperiod>=2 且 1<=fastperiod<slowperiod")
    if not isinstance(KAMA_DIFF_DAYS, (int, np.integer)) or KAMA_DIFF_DAYS < 1:
        raise ValueError("KAMA_DIFF_DAYS 必须是正整数")


def kama_custom(prices, timeperiod, fastperiod, slowperiod):
    """按效率比和指定快/慢平滑周期计算 KAMA；首次取前一收盘价作种子。"""
    series = pd.Series(prices, dtype=float)
    if timeperiod < 2 or not 1 <= fastperiod < slowperiod:
        raise ValueError("Invalid KAMA periods")
    if len(series) <= timeperiod:
        return pd.Series(np.nan, index=series.index)
    if not np.isfinite(series.to_numpy()).all() or series.le(0).any():
        raise ValueError("KAMA prices must be finite and positive")
    change = series.diff(timeperiod).abs()
    volatility = series.diff().abs().rolling(timeperiod).sum()
    er = change.div(volatility.where(volatility > 0)).fillna(0).clip(0, 1)
    fast_sc, slow_sc = 2 / (fastperiod + 1), 2 / (slowperiod + 1)
    smoothing = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    values = np.full(len(series), np.nan)
    start = timeperiod
    values[start] = float(series.iloc[start - 1]) + float(smoothing.iloc[start]) * (
        float(series.iloc[start]) - float(series.iloc[start - 1]))
    for i in range(start + 1, len(series)):
        values[i] = values[i - 1] + float(smoothing.iloc[i]) * (
            float(series.iloc[i]) - values[i - 1])
    return pd.Series(values, index=series.index)


def kama_filter(history):
    """Signal computed solely from closes known at decision time."""
    records = {}
    for ticker in history.columns:
        price = history[ticker]
        fast = kama_custom(price, *KAMA_FAST)
        slow = kama_custom(price, *KAMA_SLOW)
        diff = fast.diff(KAMA_DIFF_DAYS)
        records[ticker] = {
            "kama_fast": fast.iloc[-1], "kama_slow": slow.iloc[-1],
            "kama_fast_diff": diff.iloc[-1],
            "kama_eligible": bool(pd.notna(slow.iloc[-1]) and pd.notna(diff.iloc[-1])
                                  and fast.iloc[-1] > slow.iloc[-1] and diff.iloc[-1] > 0)
        }
    return pd.DataFrame.from_dict(records, orient="index")


def precompute_kama(close):
    """Compute each ETF KAMA once; signals are indexed by known closing dates."""
    pieces = {}
    for ticker in close.columns:
        price = close[ticker]
        fast = kama_custom(price, *KAMA_FAST)
        slow = kama_custom(price, *KAMA_SLOW)
        diff = fast.diff(KAMA_DIFF_DAYS)
        pieces[ticker] = pd.DataFrame({
            "kama_fast": fast, "kama_slow": slow,
            "kama_fast_diff": diff,
            "kama_eligible": (fast > slow) & (diff > 0),
        }, index=close.index)
    return pieces


def make_momentum_signal(method, windows=None, min_positive=None, use_kama=None, kama_cache=None, momentum_cache=None):
    """窗口只影响选资产；EPO 仍采用原收益均值/协方差与收缩公式。"""
    windows, required = momentum_settings(windows, min_positive)
    use_kama = USE_KAMA_FILTER if use_kama is None else use_kama
    if method not in {"equal", "epo"}:
        raise ValueError("Unknown momentum allocation method")
    score_records, decision_records = [], []
    selected_counts, status_counts = {}, {}

    def signal_fn(date, history):
        if history.empty or history.index[-1] != pd.Timestamp(date):
            raise ValueError("History must end exactly at signal close")
        key = (pd.Timestamp(date), tuple(windows), required)
        if momentum_cache is not None and key in momentum_cache:
            scores = momentum_cache[key].copy()
        else:
            scores = momentum_scores(history, windows, required)
            if momentum_cache is not None:
                momentum_cache[key] = scores.copy()
        if use_kama:
            if kama_cache is None:
                kama = kama_filter(history)
            else:
                kama = pd.DataFrame.from_dict(
                    {t: {col: frame.at[pd.Timestamp(date), col]
                         for col in ("kama_fast", "kama_slow", "kama_fast_diff", "kama_eligible")}
                     for t, frame in kama_cache.items()}, orient="index")
            scores = scores.join(kama)
            scores["eligible"] = scores["eligible"] & scores["kama_eligible"]
            if not KAMA_FILTER_DEFENSIVE:
                scores.loc[DEFENSIVE_ETF, "eligible"] = False
        selected = scores.index[scores.eligible].tolist()[:TOP_N]
        if not selected:
            if DEFENSIVE_ETF not in history.columns:
                raise ValueError("Defensive ETF missing from universe")
            target = pd.Series({DEFENSIVE_ETF: 1.0})
            details = {"status": "defensive", "reason": "no_eligible_candidate"}
        elif method == "equal":
            target = pd.Series(1 / len(selected), index=selected)
            details = {"status": "equal"}
        else:
            target, details = epo_weights(history, selected)
        for ticker in selected:
            selected_counts[ticker] = selected_counts.get(ticker, 0) + 1
        status = details["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if EXPORT_DETAILS:
            for ticker, row in scores.iterrows():
                score_records.append({"signal_close": date, "ticker": ticker,
                                      **row.to_dict(), "selected": ticker in selected,
                                      "target_weight": float(target.get(ticker, 0))})
            serial_details = {k: v.tolist() if isinstance(v, np.ndarray) else v
                              for k, v in details.items()}
            decision_records.append({"signal_close": date,
                                     "windows": "|".join(map(str, windows)),
                                     "required_positive_windows": required,
                                     "selected": "|".join(selected),
                                     "status": status, "reason": details.get("reason", ""),
                                     "target_weights": json.dumps(target.to_dict()),
                                     "epo_details": json.dumps(serial_details)})
        return target.to_dict()

    signal_fn.score_records = score_records
    signal_fn.decision_records = decision_records
    signal_fn.selected_counts = selected_counts
    signal_fn.status_counts = status_counts
    signal_fn.windows = windows
    signal_fn.required_positive_windows = required
    return signal_fn


def print_epo_example():
    """两资产例子，直接调用与实际策略相同的公式，使用一致的年化单位。"""
    s = np.array([0.08, 0.04])
    cov = np.array([[0.04, 0.01], [0.01, 0.01]])
    weights, details = epo_from_moments(s, cov, 0.2)
    print("Illustration only: A/B returns=8%/4%, vol=20%/10%, correlation=0.5")
    for key, value in details.items():
        print(f"{key}:\n{value}")
    print(f"Final allocation: A={weights[0]:.4%}, B={weights[1]:.4%}")


# ==================== 4. Self-financing trade accounting ====================
def trade(equity, holdings, target, cost_bps=None):
    """按交易费后的净值定目标权重，并将剩余净值记为现金（可为负）。"""
    k = (COST_BPS if cost_bps is None else cost_bps) / 10000
    post = equity
    for _ in range(100):
        new_post = equity - k * np.abs(target * post - holdings).sum()
        if abs(new_post - post) < equity * 1e-12:
            post = new_post
            break
        post = new_post
    new_holdings = target * post
    traded = np.abs(new_holdings - holdings).sum()
    fee = k * traded
    cash = equity - fee - new_holdings.sum()
    return new_holdings, cash, fee, traded / equity


def leveraged_equal_weight_target(tickers, leverage, defensive=DEFENSIVE_ETF):
    """可选基准只对非防御资产等权配仓，不持有 BIL。"""
    if defensive not in tickers:
        raise ValueError("Defensive ETF missing from leveraged benchmark")
    risky = [ticker for ticker in tickers if ticker != defensive]
    if not risky:
        raise ValueError("Leveraged benchmark needs a non-defensive ETF")
    return {ticker: leverage / len(risky) for ticker in risky}


# ==================== 5. Backtest engine ====================
def backtest(close, adj_open, name, signal_fn,
             rebalance="monthly", gross_cap=None, audit=None, cost_bps=None):
    dates, columns = adj_open.index, list(adj_open.columns)
    if not close.index.equals(dates) or list(close.columns) != columns:
        raise ValueError("Close and open panels must have identical dates/columns")
    if rebalance not in {"monthly", "buy_hold"}:
        raise ValueError("Unknown rebalance schedule")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = dates.searchsorted(pd.Timestamp(BACKTEST_START))
    if start < 1 or start >= len(dates) - 1:
        raise ValueError("Need prior-close history and at least two OOS opening prices")
    holdings, cash, equity = np.zeros(len(columns)), 1.0, 1.0
    rows, weight_rows, trade_rows = [], [], []
    pnl_rows, fee_rows, holding_rows, return_rows = [], [], [], []
    effective_cost = COST_BPS if cost_bps is None else float(cost_bps)
    if not 0 <= effective_cost <= 100:
        raise ValueError("Invalid backtest transaction cost")
    previous_month = None
    for d in range(start, len(dates) - 1):
        initial = d == start
        month = (dates[d].year, dates[d].month)
        scheduled = initial or (rebalance == "monthly" and month != previous_month)
        current = holdings / equity
        repair = gross_cap is not None and current.sum() > gross_cap + 1e-9
        fee, turnover = 0.0, 0.0
        asset_fees = np.zeros(len(columns))
        if scheduled or repair:
            if scheduled:
                # Callback only sees prices available by the previous close.
                target = signal_fn(dates[d - 1], close.iloc[:d].copy())
                target = pd.Series(target).reindex(columns).fillna(0).to_numpy(float)
                if not np.isfinite(target).all() or (target < 0).any():
                    raise ValueError(f"{name}: invalid target")
                if gross_cap is not None and target.sum() > gross_cap + 1e-9:
                    raise ValueError(f"{name}: target exceeds leverage cap")
                if (name.startswith("Momentum_") or name == "Momentum_kama") and not np.isclose(target.sum(), 1):
                    raise ValueError(f"{name}: momentum weights must sum to one")
            else:
                # Repair opening leverage drift by reducing existing positions.
                target = current * gross_cap / current.sum()
            previous_holdings = holdings.copy()
            holdings, cash, fee, turnover = trade(equity, holdings, target, effective_cost)
            asset_fees = effective_cost / 10000 * np.abs(holdings - previous_holdings)
            trade_rows.append({"execution_open": dates[d],
                               "signal_close": dates[d - 1] if scheduled else pd.NaT,
                               "scheduled": scheduled, "risk_repair": repair,
                               "fee": fee, "turnover": turnover,
                               **{t: float(w) for t, w in zip(columns, target)}})
        post_equity = equity - fee
        weights = holdings / post_equity
        if gross_cap is not None and weights.sum() > gross_cap + 1e-8:
            raise AssertionError("Post-trade leverage exceeds cap")
        days = (dates[d + 1] - dates[d]).days
        annual_rate = CASH_RATE + (BORROW_SPREAD if cash < 0 else 0)
        cash_pnl = cash * ((1 + annual_rate) ** (days / 365) - 1)
        returns = adj_open.iloc[d + 1].to_numpy() / adj_open.iloc[d].to_numpy() - 1
        individual_pnl = holdings * returns
        asset_pnl = individual_pnl.sum()
        end_equity = post_equity + asset_pnl + cash_pnl
        if not np.isfinite(end_equity) or end_equity <= 0:
            raise RuntimeError(f"{name}: equity <= 0 on {dates[d + 1]}")
        rows.append({"end_open": dates[d + 1], "start_open": dates[d],
                     "signal_close": dates[d - 1], "equity_start": equity,
                     "NAV": end_equity, "return": end_equity / equity - 1,
                     "asset_pnl": asset_pnl, "cash_pnl": cash_pnl, "fee": fee,
                     "turnover": turnover, "gross": weights.sum(),
                     "cash_weight": cash / post_equity, "risk_repair": repair,
                     "scheduled_rebalance": scheduled,
                     "rf": (1 + CASH_RATE) ** (days / 365) - 1})
        weight_rows.append(dict(zip(columns, weights), Date=dates[d]))
        if audit is not None:
            for container, values in [(pnl_rows, individual_pnl), (fee_rows, asset_fees),
                                      (holding_rows, holdings.copy()), (return_rows, returns)]:
                container.append(dict(zip(columns, values), end_open=dates[d + 1]))
        holdings *= 1 + returns
        cash += cash_pnl
        equity = end_equity
        previous_month = month
    ledger = pd.DataFrame(rows).set_index("end_open")
    weights = pd.DataFrame(weight_rows).set_index("Date")
    if EXPORT_DETAILS:
        folder = OUT_DIR / "details"
        folder.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(folder / f"{name}_ledger.csv")
        weights.to_csv(folder / f"{name}_weights.csv")
        pd.DataFrame(trade_rows).to_csv(folder / f"{name}_trades.csv", index=False)
    if audit is not None:
        audit.update(weights=weights, trades=pd.DataFrame(trade_rows))
        for key, values in [("asset_pnl", pnl_rows), ("asset_fees", fee_rows),
                            ("holding_values", holding_rows), ("asset_returns", return_rows)]:
            audit[key] = pd.DataFrame(values).set_index("end_open")
        reconstructed = (audit["asset_pnl"].sum(axis=1) + ledger.cash_pnl - ledger.fee)
        if (not np.allclose(reconstructed, ledger.NAV-ledger.equity_start,
                            atol=1e-10, rtol=1e-9)
                or not np.allclose(audit["asset_fees"].sum(axis=1), ledger.fee,
                                   atol=1e-12, rtol=1e-10)):
            raise AssertionError("Attribution does not reconcile with portfolio NAV")
    return ledger


# ==================== 6. Performance and charts ====================
def highlight_strategy_lines(ax, ncol=1, fontsize=9):
    """用于净值与回撤两类图：突出策略，弱化背景，图例策略排在前面。"""
    benchmark_styles = {
        "SPY": {"color": "#222222", "linestyle": "-.", "linewidth": 1.6},
        "QQQ": {"color": "#777777", "linestyle": "--", "linewidth": 1.5},
        "EqualWeight": {"color": "#666666", "linestyle": ":", "linewidth": 1.6},
        "BIL": {"color": "#999999", "linestyle": "--", "linewidth": 1.2},
        "LeveragedEqualWeight": {"color": "#555555", "linestyle": "--", "linewidth": 1.4},
        "Momentum_EPO_34D": {"color": "#9467BD", "linestyle": ":", "linewidth": 1.8},
    }
    for line in ax.get_lines():
        name = line.get_label()
        if name in STRATEGY_PLOT_STYLES:
            line.set(**STRATEGY_PLOT_STYLES[name])
        else:
            line.set(color="#AAAAAA", linewidth=0.9, alpha=0.55, zorder=2)
            if name in benchmark_styles:
                line.set(**benchmark_styles[name], alpha=0.85, zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    priority = {"Momentum_kama": 0, "Momentum_EQ": 1, "SPY": 2, "QQQ": 3, "EqualWeight": 4}
    order = sorted(range(len(labels)), key=lambda i: priority.get(labels[i], 5))
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                       ncol=ncol, fontsize=fontsize, handlelength=3.5)
    for label in legend.get_texts():
        if label.get_text() in STRATEGY_PLOT_STYLES:
            label.set_fontweight("bold")


def summarize(name, ledger):
    r = ledger["return"]
    excess = r - ledger.rf
    years = (ledger.index[-1] - ledger.start_open.iloc[0]).days / 365.25
    nav = np.r_[1.0, ledger.NAV.to_numpy()]
    max_dd = (nav / np.maximum.accumulate(nav) - 1).min()
    sd = excess.std(ddof=1)
    return {"Strategy": name, "CAGR": ledger.NAV.iloc[-1] ** (1 / years) - 1,
            "Volatility": r.std(ddof=1) * np.sqrt(252),
            "Sharpe": excess.mean() / sd * np.sqrt(252) if sd > 1e-12 else np.nan,
            "MaxDrawdown": max_dd, "MeanExposure": ledger.gross.mean(),
            "AnnualTurnover": ledger.turnover.iloc[1:].sum() / years}


def drawdown_episodes(ledger):
    """以开盘净值识别回撤峰值、谷底、恢复日期；初始净值也纳入。"""
    nav = pd.concat([pd.Series([ledger.equity_start.iloc[0]],
                               index=[ledger.start_open.iloc[0]]), ledger.NAV])
    peak_date, peak = nav.index[0], float(nav.iloc[0])
    active, records = None, []
    for date, value in nav.iloc[1:].items():
        if value >= peak:
            if active is not None:
                active.update(recovery_date=date, recovered=True,
                              underwater_calendar_days=(date-active["peak_date"]).days)
                records.append(active)
                active = None
            peak_date, peak = date, float(value)
        else:
            depth = float(value / peak - 1)
            if active is None:
                active = {"peak_date": peak_date, "peak_nav": peak,
                          "trough_date": date, "trough_nav": float(value),
                          "drawdown": depth}
            elif depth < active["drawdown"]:
                active.update(trough_date=date, trough_nav=float(value), drawdown=depth)
    if active is not None:
        active.update(recovery_date=pd.NaT, recovered=False,
                      underwater_calendar_days=(nav.index[-1]-active["peak_date"]).days)
        records.append(active)
    columns = ["peak_date", "peak_nav", "trough_date", "trough_nav", "drawdown",
               "recovery_date", "recovered", "underwater_calendar_days"]
    return pd.DataFrame(records, columns=columns).sort_values("drawdown")


def period_attribution(ledger, audit, selected_rows):
    """真实持仓盈亏 / 该期初始净资产：精确加总为该期累计收益。

    pnl 已经包含前面净值变化带来的持仓金额变化，因此不能再次复利。
    FeeContribution 为负；NetContribution = GrossContribution + FeeContribution。
    贡献不是资产自身收益率，也不是 CAGR 的分解。
    """
    part = ledger.loc[selected_rows]
    if part.empty:
        return None, None
    initial, final = float(part.equity_start.iloc[0]), float(part.NAV.iloc[-1])
    gross = audit["asset_pnl"].loc[part.index].sum()
    fees = audit["asset_fees"].loc[part.index].sum()
    components = pd.DataFrame({"GrossPnL": gross, "FeesPaid": fees})
    components.loc["CASH_INTEREST"] = [part.cash_pnl.sum(), 0.0]
    components["NetPnL"] = components.GrossPnL - components.FeesPaid
    components["GrossContribution"] = components.GrossPnL / initial
    components["FeeContribution"] = -components.FeesPaid / initial
    components["NetContribution"] = components.NetPnL / initial
    reconstructed = float(components.NetPnL.sum())
    error = reconstructed - (final-initial)
    if not np.isclose(reconstructed, final-initial, atol=1e-10, rtol=1e-9):
        raise AssertionError("Period attribution does not reconcile")
    summary = {"start_open": part.start_open.iloc[0], "end_open": part.index[-1],
               "start_NAV": initial, "end_NAV": final,
               "period_return": final/initial-1,
               "contribution_sum": components.NetContribution.sum(),
               "fees_NAV_units": part.fee.sum(),
               "fees_over_period_start_NAV": part.fee.sum()/initial,
               "reconciliation_error_NAV_units": error}
    return components, summary


def export_attribution(name, ledger, audit, signal):
    """构建全期、关键阶段和最大回撤段归因；逐日输出由 EXPORT_DETAILS 控制。"""
    masks = {"Full": np.ones(len(ledger), dtype=bool)}
    for label, (start, end) in ATTRIBUTION_PERIODS.items():
        masks[label] = (ledger.index >= pd.Timestamp(start)) & (ledger.index < pd.Timestamp(end))
    episodes = drawdown_episodes(ledger)
    if not episodes.empty:
        worst = episodes.iloc[0]
        masks["Worst_drawdown_peak_to_trough"] = ((ledger.index > worst.peak_date)
                                                  & (ledger.index <= worst.trough_date))
    tables, summaries = {}, []
    for label, mask in masks.items():
        table, summary = period_attribution(ledger, audit, mask)
        if table is not None:
            table.index.name = "Component"
            tables[label] = table
            summaries.append({"Period": label, **summary})
    weights = audit["weights"].copy()
    weights.index = ledger.index
    held = weights.gt(1e-10)
    pnl, fees = audit["asset_pnl"], audit["asset_fees"]
    selected_counts = pd.Series(signal.selected_counts, dtype=float)
    holdings = pd.DataFrame({
        "MeanWeight": weights.mean(), "MaxWeight": weights.max(),
        "MeanWeightWhenHeld": weights.where(held).mean(),
        "HeldIntervals": held.sum(), "HeldFraction": held.mean(),
        "SelectedRebalances": selected_counts.reindex(weights.columns).fillna(0).astype(int),
        "GrossPnL": pnl.sum(), "FeesPaid": fees.sum(), "NetPnL": (pnl-fees).sum()})
    holdings.index.name = "ETF"
    exposure = pd.DataFrame(index=ledger.index)
    exposure["MaxAssetWeight"] = weights.max(axis=1)
    exposure["HeldAssetCount"] = held.sum(axis=1)
    # 这是 ETF 组代理敞口，不穿透成分股。
    for group, members in ASSET_GROUPS.items():
        exposure[group] = weights[[t for t in members if t in weights]].sum(axis=1)
    if EXPORT_DETAILS:
        folder = OUT_DIR / "details" / name
        folder.mkdir(parents=True, exist_ok=True)
        for key in ["asset_pnl", "asset_fees", "holding_values", "asset_returns"]:
            audit[key].to_csv(folder / f"{key}_daily.csv", encoding="utf-8-sig")
        exposure.to_csv(folder / "exposure_daily.csv", encoding="utf-8-sig")
        episodes.to_csv(folder / "drawdown_episodes.csv", index=False, encoding="utf-8-sig")
    return {"tables": tables, "summary": pd.DataFrame(summaries),
            "holdings": holdings, "exposure": exposure, "episodes": episodes,
            "status_counts": signal.status_counts.copy()}


def compare_attribution(reports):
    """合并两条策略的归因，并比较相同日历区间的净贡献。"""
    full_rows, period_rows, holding_rows = [], [], []
    for name, report in reports.items():
        for period, table in report["tables"].items():
            rows = table.reset_index().assign(Strategy=name, Period=period)
            period_rows.append(rows)
            if period == "Full":
                full_rows.append(rows.drop(columns="Period"))
        holding_rows.append(report["holdings"].reset_index().assign(Strategy=name))
    full = pd.concat(full_rows, ignore_index=True).sort_values(
        ["Strategy", "NetContribution"], ascending=[True, False])
    full.to_csv(OUT_DIR / "asset_contribution_full.csv", index=False, encoding="utf-8-sig")
    pd.concat(period_rows, ignore_index=True).to_csv(
        OUT_DIR / "contribution_by_period.csv", index=False, encoding="utf-8-sig")
    pd.concat(holding_rows, ignore_index=True).to_csv(
        OUT_DIR / "holding_summary.csv", index=False, encoding="utf-8-sig")
    epo, equal = reports["Momentum_kama"], reports["Momentum_EQ"]
    comparisons = []
    for period, left in epo["tables"].items():
        # 最大回撤段的日期不同，不能直接比较。
        if period == "Worst_drawdown_peak_to_trough" or period not in equal["tables"]:
            continue
        right = equal["tables"][period]
        frame = pd.DataFrame({"EPO_NetContribution": left.NetContribution,
                              "EQ_NetContribution": right.NetContribution})
        frame["NetDifference"] = frame.EPO_NetContribution - frame.EQ_NetContribution
        frame.index.name = "Component"
        comparisons.append(frame.reset_index().assign(Period=period))
    comparison = pd.concat(comparisons, ignore_index=True).sort_values(
        ["Period", "NetDifference"], ascending=[True, False])
    comparison.to_csv(OUT_DIR / "EPO_vs_EQ_contribution.csv", index=False, encoding="utf-8-sig")
    # 两条策略同时改变 KAMA 筛选和配权方法，差额不能解释为 EPO 的单独贡献。
    # 全期净贡献图同时使用颜色和纹理区分两条策略。
    frame = pd.DataFrame({"Momentum_kama": epo["tables"]["Full"].NetContribution,
                          "Momentum_EQ": equal["tables"]["Full"].NetContribution}) * 100
    frame = frame.loc[frame.Momentum_kama.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(13, 6))
    frame.plot.bar(ax=ax, color=["#0072B2", "#E69F00"], width=.8)
    for patch in ax.containers[1]:
        patch.set_hatch("//")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set(title="Full-period net contribution", xlabel="",
           ylabel="Contribution (percentage points of initial NAV)")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "contribution_comparison.png", dpi=160)
    plt.close(fig)
    return comparison


def export_cost_analysis(results, zero_cost_results):
    """费用影响只留一张表；零费用收益来自真正重跑，保留利息假设。"""
    rows = []
    for name in ["Momentum_kama", "Momentum_EQ", "Momentum_EPO_34D"]:
        ledger = results[name]
        years = (ledger.index[-1] - ledger.start_open.iloc[0]).days / 365.25
        row = {"Strategy": name, "NetCAGR": summarize(name, ledger)["CAGR"],
               "NetFinalNAV": ledger.NAV.iloc[-1],
               "TotalFees_NAV_units": ledger.fee.sum(),
               "AnnualizedFeeRatio": (ledger.fee / ledger.equity_start).sum() / years,
               "AnnualTurnoverExInitial": ledger.turnover.iloc[1:].sum() / years}
        if name in zero_cost_results:
            gross = zero_cost_results[name]
            row["ZeroCostCAGR"] = summarize(name, gross)["CAGR"]
            row["ZeroCostFinalNAV"] = gross.NAV.iloc[-1]
            row["CAGRDrag"] = row["ZeroCostCAGR"] - row["NetCAGR"]
        rows.append(row)
    costs = pd.DataFrame(rows).set_index("Strategy")
    costs.to_csv(OUT_DIR / "cost_summary.csv", encoding="utf-8-sig")
    return costs


def markdown_table(frame):
    """生成 Markdown 表，不要求额外安装 tabulate。"""
    if frame.empty:
        return "本次数据范围内无记录。"
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ")
                                     for v in row) + " |")
    return "\n".join(lines)


def write_report(performance, yearly, reports, costs, results):
    """一份中文摘要：绩效、阶段、归因、USO、集中度和费用。"""
    fmt_pct = lambda x: f"{x:.2%}" if pd.notna(x) else "—"
    fmt_pp = lambda x: f"{100*x:+.2f}" if pd.notna(x) else "—"
    preferred = ["Momentum_kama", "Momentum_EQ", "Momentum_EPO_34D", "SPY", "QQQ", "EqualWeight",
                 "LeveragedEqualWeight", "BIL"]
    selected = [n for n in preferred if n in performance.index]
    perf = performance.loc[selected, ["CAGR", "Volatility", "Sharpe", "MaxDrawdown",
                                      "AnnualTurnover"]].copy()
    for col in ["CAGR", "Volatility", "MaxDrawdown"]:
        perf[col] = perf[col].map(fmt_pct)
    perf["Sharpe"] = perf.Sharpe.map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    perf["AnnualTurnover"] = perf.AnnualTurnover.map(lambda x: f"{x:.2f} 倍")
    perf = perf.rename(columns={"CAGR": "年化收益", "Volatility": "年化波动",
                               "Sharpe": "Sharpe", "MaxDrawdown": "最大回撤",
                               "AnnualTurnover": "年化买卖交易量"})
    perf.index.name = "策略"
    ledger = results["Momentum_kama"]
    windows, required = momentum_settings()
    lines = ["# QF5210 ETF 策略回测：动量、KAMA 与 EPO", "",
             f"实际覆盖：{ledger.start_open.iloc[0].date()} 开盘至 {ledger.index[-1].date()} 开盘。",
             f"资产池：{', '.join(TICKERS)}。单边交易成本：{COST_BPS:g} bp。",
             f"Momentum_kama 的 KAMA 筛选：{USE_KAMA_FILTER}；快线{KAMA_FAST}，慢线{KAMA_SLOW}，快线{KAMA_DIFF_DAYS}日变化>0；BIL正常入选需通过筛选：{KAMA_FILTER_DEFENSIVE}。",
             f"动量窗口：{windows} 个收盘价格；最多持有 {TOP_N} 只；每月首个交易日开盘调仓，信号使用上一交易日收盘。",
             f"窗口内分数：expm1({MOMENTUM_ANNUALIZATION} × log-price 回归斜率) × R²。",
             f"单窗口按原分数排序；多窗口分别排名后取平均名次，越小越优。要求至少 {required}/{len(windows)} 个窗口分数为正，且所有窗口均有有效数据。",
             "多窗口 score=1/平均名次，仅作排序；筛选看 eligible，不是看合成 score 是否大于零。",
             f"EPO：{EPO_PRICE_DAYS} 个价格估计日收益均值/协方差，收缩参数 {EPO_SHRINKAGE:g}，非负截断后归一化。",
             ("资产池不含 TLT、TIP；固定收益 ETF 仅保留 BIL。无合格候选时满仓 BIL；"
              + ("正常入选时 BIL 必须通过动量与 KAMA。" if USE_KAMA_FILTER and KAMA_FILTER_DEFENSIVE
                 else "KAMA 开启时 BIL 不正常入选，仅作回退。" if USE_KAMA_FILTER
                 else "KAMA 关闭时 BIL 与其他 ETF 一样按动量筛选。")),
             "Momentum_EQ、Momentum_kama 和 Momentum_EPO_34D 的目标 ETF 权重均合计 100%，不使用账户杠杆、目标波动率或周度仓位控制。",
             "Momentum_EQ 按动量入选后等权、不使用 KAMA；Momentum_kama 按动量与可选 KAMA 筛选后 EPO 配权；Momentum_EPO_34D 是固定 34 日动量、不使用 KAMA 的 EPO 对照。",
             ("当前动量窗口为 [34]；KAMA 开启时 Momentum_kama 与 34 日 EPO 对照仍可能不同，差异包括 KAMA 筛选及其后续持仓。"
              if windows == [34] else "当前为多窗口实验；Momentum_kama 与 34 日 EPO 对照还同时存在动量窗口和 KAMA 筛选差异。"),
             (f"杠杆等权基准：启用={RUN_LEVERAGED_EW}，杠杆上限={MAX_LEVERAGE:g}；只买非 BIL 资产，借款利率为现金利率加融资利差。"
              if RUN_LEVERAGED_EW else "杠杆等权基准未运行；MAX_LEVERAGE 不影响上述三条动量策略。"),
             "删去资产与窗口选择均已使用历史表现作开发判断，未来结果仍需固定规则向前评估。", "",
             "## 绩效", "", markdown_table(perf.reset_index()), "",
             "年化买卖交易量是买入金额与卖出金额分别相对交易前净值的总和，排除首次建仓；不是除以二的单边换手。", "",
             "## 年度收益", ""]
    annual = yearly[[n for n in ["Momentum_kama", "Momentum_EQ", "Momentum_EPO_34D", "SPY", "EqualWeight"]
                     if n in yearly]].copy()
    for col in annual:
        annual[col] = annual[col].map(fmt_pct)
    annual.index.name = "年份"
    lines += [markdown_table(annual.reset_index()), "", "首尾年份可能不完整；各年收益按持有区间结束日期归属。", ""]
    if reports:
        epo, equal = reports["Momentum_kama"], reports["Momentum_EQ"]
        left, right = epo["tables"]["Full"], equal["tables"]["Full"]
        full = pd.DataFrame({"KAMA与EPO净贡献": left.NetContribution,
                             "等权净贡献": right.NetContribution})
        full["差额"] = full["KAMA与EPO净贡献"] - full["等权净贡献"]
        full = full.sort_values("KAMA与EPO净贡献", ascending=False)
        full.loc["合计"] = full.sum()
        full = full.map(fmt_pp)
        full.index.name = "资产"
        lines += ["## 全期净收益归因", "", markdown_table(full.reset_index()), "",
                  "单位：累计收益百分点；贡献已经扣除该资产交易费用。两条策略同时改变 KAMA 筛选和配权，差额不能归因于 EPO 单一因素，也不是 CAGR 的分解。", ""]
        summaries = {n: r["summary"].set_index("Period") for n, r in reports.items()}
        stage_rows = []
        for period in ATTRIBUTION_PERIODS:
            if period not in summaries["Momentum_kama"].index:
                continue
            a, b = summaries["Momentum_kama"].loc[period], summaries["Momentum_EQ"].loc[period]
            stage_rows.append({"阶段": period,
                               "实际开始开盘": pd.Timestamp(a.start_open).date(),
                               "实际结束开盘": pd.Timestamp(a.end_open).date(),
                               "KAMA与EPO收益": fmt_pct(a.period_return), "等权收益": fmt_pct(b.period_return),
                               "差额(百分点)": fmt_pp(a.period_return - b.period_return)})
        lines += ["## 关键阶段", "", markdown_table(pd.DataFrame(stage_rows)), ""]
        episodes = epo["episodes"]
        if not episodes.empty:
            worst = episodes.iloc[0]
            losses = epo["tables"]["Worst_drawdown_peak_to_trough"].NetContribution.sort_values().head(4)
            lines += [f"Momentum_kama 最大回撤：{worst.drawdown:.2%}；峰值 {worst.peak_date.date()} → 谷底 {worst.trough_date.date()}。",
                      "该段净贡献最低的四项：" + "；".join(f"{t} {100*v:+.2f} 个百分点" for t, v in losses.items()) + "。", ""]
        lines += ["## 持仓与集中度", ""]
        risk_rows = []
        for name, report in reports.items():
            exposure = report["exposure"]
            risk_rows.append({"策略": name, "最大单资产权重": fmt_pct(exposure.MaxAssetWeight.max()),
                              "平均实际持有数": f"{exposure.HeldAssetCount.mean():.2f}",
                              "平均QQQ/XLK/SMH权重": fmt_pct(exposure.Tech_related_ETFs.mean()),
                              "EPO退回等权次数": report["status_counts"].get("fallback_equal", 0)
                              if name == "Momentum_kama" else "—"})
        lines += [markdown_table(pd.DataFrame(risk_rows)), "",
                  "科技敞口只是 QQQ、XLK、SMH 的 ETF 权重合计，未穿透成分股；入选三只不代表实际持有三只。", ""]
        if "USO" in epo["holdings"].index:
            holding = epo["holdings"].loc["USO"]
            lines += ["## USO", "",
                      f"Momentum_kama：全期净贡献 {100*left.loc['USO', 'NetContribution']:+.2f} 个百分点；平均权重 {holding.MeanWeight:.2%}；实际持有区间占比 {holding.HeldFraction:.2%}；最大权重 {holding.MaxWeight:.2%}。",
                      f"动量等权：全期 USO 净贡献 {100*right.loc['USO', 'NetContribution']:+.2f} 个百分点。",
                      "这是实际持仓贡献；量化加入 USO 的影响仍需固定其他设置，单独跑不含 USO 的对照。", ""]
    cost_display = costs[["NetCAGR", "AnnualTurnoverExInitial", "AnnualizedFeeRatio"]].copy()
    for col in ["ZeroCostCAGR", "CAGRDrag"]:
        if col in costs:
            cost_display[col] = costs[col]
    for col in cost_display:
        cost_display[col] = cost_display[col].map(
            (lambda x: f"{x:.2f} 倍") if col == "AnnualTurnoverExInitial"
            else fmt_pp if col == "CAGRDrag" else fmt_pct)
    cost_display = cost_display.rename(columns={
        "NetCAGR": "扣费年化收益", "AnnualTurnoverExInitial": "年化买卖交易量",
        "AnnualizedFeeRatio": "年均费用/交易前净值", "ZeroCostCAGR": "零费用年化收益",
        "CAGRDrag": "费用拖累(百分点/年)"})
    cost_display.index.name = "策略"
    lines += ["## 交易费用", "", markdown_table(cost_display.reset_index()), "",
              "零费用结果是重新回测后的复利结果，仍计现金/融资利息；费用拖累不能用累计已付费用直接代替。", "",
              "## 文件与口径", "",
              "performance.csv：全部 ETF 和组合绩效；nav.csv：全部净值；yearly_returns.csv：年度收益；cost_summary.csv：费用影响。",
              "asset_contribution_full.csv：全期归因；contribution_by_period.csv：全期、关键阶段和各策略自身最大回撤段；holding_summary.csv：持仓。三张表都用 Strategy 列区分策略；EPO_vs_EQ_contribution.csv 以历史文件名比较 Momentum_kama 与 Momentum_EQ，差异不只来自 EPO。",
              "backtest.png：全部净值/回撤；momentum_comparison.png：重点策略净值/回撤；contribution_comparison.png：两条策略的全期贡献。",
              "CSV 的收益、贡献、权重保存为小数，0.05=5%；PnL/FeesPaid 为初始净值=1 的金额单位。各期贡献分母不同，不能跨期直接相加。费用已扣入净贡献，不能重复扣除。",
              "仅保留四类核心检查：价格/日期有效；信号早于成交；仓位有限且符合满仓约定；净值、持仓盈亏和费用对账。详细账本只在 EXPORT_DETAILS=True 时保存到 details/。",
              f"本次 EXPORT_DETAILS={EXPORT_DETAILS}；scores.csv 保存各窗口原始分数、窗口排名、average_rank、positive_windows、eligible 和 selected；decisions.csv 保存实际窗口配置与目标权重。",
              "模型假设：复权开盘是总收益近似；现金/融资使用常数利率；只收显式交易成本，无其他市场冲击模型。本结果是历史回测，已看过或用于调参的区间不能视为全新样本外。", ""]
    if not reports:
        lines.append("RUN_ATTRIBUTION=False：本次未生成归因表和贡献图。")
    if "ZeroCostCAGR" not in costs:
        lines.append("RUN_COST_COUNTERFACTUAL=False：本次未重跑零费用情景，不能报告费用的 CAGR 拖累。")
    (OUT_DIR / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    validate_kama_parameters()
    if not 1 <= MAX_LEVERAGE <= 2:
        raise ValueError("MAX_LEVERAGE must be in [1, 2]")
    if not 0 <= COST_BPS <= 100 or CASH_RATE <= -1 or BORROW_SPREAD < 0:
        raise ValueError("Invalid cost/rate assumptions")
    windows, required = momentum_settings()
    if TOP_N < 1 or EPO_PRICE_DAYS < 3:
        raise ValueError("Invalid momentum/risk lookback or TOP_N")
    if not 0 <= EPO_SHRINKAGE <= 1:
        raise ValueError("EPO_SHRINKAGE must be in [0, 1]")
    if DEFENSIVE_ETF not in TICKERS or len(set(TICKERS)) != len(TICKERS):
        raise ValueError("Universe must contain defensive ETF and no duplicates")
    print(f"Universe ({len(TICKERS)}): {TICKERS}")
    print(f"Momentum_EQ: no KAMA; Momentum_kama: KAMA={USE_KAMA_FILTER} + EPO; "
          "monthly, long-only, target gross=1.0")
    print(f"Momentum windows={windows}; require {required}/{len(windows)} positive windows")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {k: globals()[k] for k in [
        "TICKERS", "USE_EXPANDED", "DATA_START", "DATA_END", "BACKTEST_START",
        "COST_BPS", "CASH_RATE", "BORROW_SPREAD", "MAX_LEVERAGE", "RUN_LEVERAGED_EW",
        "RUN_ATTRIBUTION", "RUN_COST_COUNTERFACTUAL", "ATTRIBUTION_PERIODS",
        "MOMENTUM_WINDOWS", "MIN_POSITIVE_WINDOWS", "USE_KAMA_FILTER", "KAMA_FAST", "KAMA_SLOW", "KAMA_DIFF_DAYS", "KAMA_FILTER_DEFENSIVE", "MOMENTUM_ANNUALIZATION", "TOP_N", "EPO_PRICE_DAYS",
        "EPO_SHRINKAGE", "DEFENSIVE_ETF", "EXPORT_DETAILS"]}
    if EXPORT_DETAILS:
        (OUT_DIR / "details").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "details" / "config_used.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8")
    close, adj_open, volume, kama_cache = load_data()
    results, audits, signals, zero_cost_results = {}, {}, {}, {}
    momentum_cache = {}
    for ticker in TICKERS:
        results[ticker] = backtest(close, adj_open, ticker,
                                  lambda date, history, t=ticker: {t: 1.0},
                                  rebalance="buy_hold")
    ew = {t: 1 / len(TICKERS) for t in TICKERS}
    results["EqualWeight"] = backtest(close, adj_open, "EqualWeight",
                                      lambda date, history: ew)
    for name, method in [("Momentum_EQ", "equal"), ("Momentum_kama", "epo")]:
        use_kama = name == "Momentum_kama" and USE_KAMA_FILTER
        signal = make_momentum_signal(
            method, use_kama=use_kama, kama_cache=kama_cache,
            momentum_cache=momentum_cache)
        audits[name] = {}
        results[name] = backtest(close, adj_open, name, signal,
                                 rebalance="monthly", gross_cap=1.0,
                                 audit=audits[name] if RUN_ATTRIBUTION else None)
        signals[name] = signal
        if EXPORT_DETAILS:
            pd.DataFrame(signal.score_records).to_csv(
                OUT_DIR / "details" / f"{name}_scores.csv", index=False)
            pd.DataFrame(signal.decision_records).to_csv(
                OUT_DIR / "details" / f"{name}_decisions.csv", index=False)
        if RUN_COST_COUNTERFACTUAL:
            zero_cost_signal = make_momentum_signal(
                method, use_kama=use_kama, kama_cache=kama_cache,
                momentum_cache=momentum_cache)
            zero_cost_results[name] = backtest(
                close, adj_open, name + "_ZeroCost", zero_cost_signal,
                rebalance="monthly", gross_cap=1.0, cost_bps=0.0)
    # 固定 34 日动量与 EPO 的对照组，不经过 KAMA 筛选。
    results["Momentum_EPO_34D"] = backtest(
        close, adj_open, "Momentum_EPO_34D", make_momentum_signal("epo", [34], 1, use_kama=False, momentum_cache=momentum_cache),
        rebalance="monthly", gross_cap=1.0)
    if RUN_COST_COUNTERFACTUAL:
        zero_cost_results["Momentum_EPO_34D"] = backtest(
            close, adj_open, "Momentum_EPO_34D_ZeroCost", make_momentum_signal("epo", [34], 1, use_kama=False, momentum_cache=momentum_cache),
            rebalance="monthly", gross_cap=1.0, cost_bps=0.0)
    if RUN_LEVERAGED_EW:
        levered = leveraged_equal_weight_target(TICKERS, MAX_LEVERAGE)
        print(f"LeveragedEqualWeight: {MAX_LEVERAGE:.2f}x across "
              f"{len(levered)} non-{DEFENSIVE_ETF} ETFs; {DEFENSIVE_ETF} weight=0")
        results["LeveragedEqualWeight"] = backtest(
            close, adj_open, "LeveragedEqualWeight", lambda date, history: levered,
            gross_cap=MAX_LEVERAGE)
    performance = pd.DataFrame([summarize(n, x) for n, x in results.items()]).set_index("Strategy")
    performance.to_csv(OUT_DIR / "performance.csv")
    names = [n for n in ["Momentum_kama", "Momentum_EQ", "Momentum_EPO_34D", "SPY", "QQQ",
                         "EqualWeight", "LeveragedEqualWeight", "BIL"] if n in performance.index]
    display = performance.loc[names].copy()
    for c in ["CAGR", "Volatility", "MaxDrawdown"]:
        display[c] = display[c].map(lambda v: f"{v:.2%}")
    print("\nPerformance (net trading costs and financing):")
    print(display.to_string(float_format=lambda v: f"{v:.3f}"))
    nav = pd.DataFrame({n: x.NAV for n, x in results.items()})
    first_open = next(iter(results.values())).start_open.iloc[0]
    nav = pd.concat([pd.DataFrame(1.0, index=[first_open], columns=nav.columns), nav])
    dd = nav / nav.cummax() - 1
    nav.to_csv(OUT_DIR / "nav.csv", index_label="Date")
    if EXPORT_DETAILS:
        dd.to_csv(OUT_DIR / "details" / "drawdown.csv", index_label="Date")
    yearly = pd.DataFrame({n: (1 + x["return"]).groupby(x.index.year).prod() - 1
                           for n, x in results.items()})
    yearly.to_csv(OUT_DIR / "yearly_returns.csv", index_label="Year")
    reports = {}
    if RUN_ATTRIBUTION:
        for name in ["Momentum_EQ", "Momentum_kama"]:
            reports[name] = export_attribution(name, results[name], audits[name], signals[name])
        compare_attribution(reports)
    costs = export_cost_analysis(results, zero_cost_results)
    write_report(performance, yearly, reports, costs, results)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    nav.plot(ax=axes[0], logy=True, title="ETF benchmarks and momentum strategies")
    axes[0].set_ylabel("NAV, initial = 1 (log scale)")
    dd.plot(ax=axes[1], title="Drawdown")
    axes[1].set_ylabel("Drawdown")
    for ax in axes:
        ax.grid(alpha=0.3)
        highlight_strategy_lines(ax, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backtest.png", dpi=160)
    print(f"\nResults saved to: {OUT_DIR}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # 单独展示动量等权、KAMA+EPO 与固定 34 日 EPO 的净值和回撤。
    comparison = [t for t in ["SPY", "QQQ", "EqualWeight", "Momentum_EQ", "Momentum_kama", "Momentum_EPO_34D", "BIL"]
                  if t in nav.columns]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    nav[comparison].plot(ax=axes[0], logy=True, title="Momentum allocation comparison")
    dd[comparison].plot(ax=axes[1], title="Momentum strategy drawdown")
    axes[0].set_ylabel("NAV, initial = 1 (log scale)")
    axes[1].set_ylabel("Drawdown")
    for ax in axes:
        ax.grid(alpha=0.3)
        highlight_strategy_lines(ax, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "momentum_comparison.png", dpi=160)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"\n全部输出完成：{OUT_DIR}\n先阅读 REPORT_CN.md；EXPORT_DETAILS={EXPORT_DETAILS}。")
    return {"results": results, "audits": audits, "reports": reports,
            "zero_cost_results": zero_cost_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF momentum, KAMA and EPO backtest")
    parser.add_argument("--epo-example", action="store_true", help="Print two-asset EPO example")
    args = parser.parse_args()
    if args.epo_example:
        print_epo_example()
    else:
        main()

