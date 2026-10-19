"""QF5210 ETF 回测：回归动量、可选 KAMA 筛选与 EPO 配权。

安装：python -m pip install numpy pandas matplotlib yfinance
运行：python QF5210_KAMA.py

先读运行后输出目录中的 REPORT_CN.md；详细输出由 EXPORT_DETAILS 控制。
不同窗口和 KAMA 参数保存在不同结果目录。
逐资产绝对净贡献 = 实际持仓盈亏 - 分摊到该资产的实际交易费用，可与组合净值对账。
主归因改用相对净贡献：实际持仓金额 ×（资产收益率 - 同期全资产等权收益率）- 交易费。
各期贡献均以期初净资产为分母，不是年化收益；绝对贡献合计等于该期累计收益。
零费用版本单独重跑，用于比较真正的费用情景差异。

新增研究诊断严格使用“调仓前一日收盘信号 → 本次调仓开盘至下次调仓开盘收益”：
报告全部 ETF 与剔除 BIL/反向 ETF 两种口径的 IC、RankIC、波动和 ICIR，
三组收益及单调性，动量乘/不乘 R² 的对照，以及 KAMA 通过组减未通过组收益。
组合指标另增加 Sortino、Calmar 和相对月度再平衡 EqualWeight 的 Information Ratio。

默认 MOMENTUM_WINDOWS=[34]，保留原回归动量；改成 [63,126,252] 即启用多周期。
单窗口按原回归分数排序；多窗口分别排名再取平均名次，越小越优。
多周期默认至少过半窗口的分数为正；合格候选最多持有 3 只，在每月首个交易日开盘调仓。
四条策略形成 2×2 对照：等权/EPO 配权 × 不使用/使用 KAMA 筛选。
Momentum_EQ、Momentum_EQ_KAMA、Momentum_EPO_NoKAMA、Momentum_kama
的目标 ETF 权重均为 100%，没有账户杠杆或目标波动率控制。
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
DATA_END = "2026-10-17"        # Exclusive: only observations strictly before this date are used.
BACKTEST_START = "2016-01-01"
COST_BPS = 10.0              # Fee/slippage assumption per dollar bought OR sold.
CASH_RATE = 0.02             # Constant scenario assumption, not historical USD rates.
BORROW_SPREAD = 0.01
MAX_LEVERAGE = 1.25          # 只用于可选的非 BIL 资产杠杆等权基准；四条动量策略上限固定为 1。
RUN_LEVERAGED_EW = False     # 默认不运行杠杆基准；打开后 BIL 权重为零。
REFRESH_DATA = False
SHOW_PLOTS = True
RUN_COST_COUNTERFACTUAL = True  # 四条动量策略分别重跑零费用情景。
RUN_ATTRIBUTION = True
EXPORT_DETAILS = True      # True 保存逐日账本、各窗口分数/排名和 EPO 输入；False 只留摘要。
RUN_SIGNAL_DIAGNOSTICS = True  # 输出 IC/RankIC、分组收益、R² 与 KAMA 检验。
SIGNAL_GROUP_COUNT = 3         # 当前完整版报告使用 High/Middle/Low 三组。
# 第二套 RankIC 口径剔除防御 ETF 和反向 ETF；若资产不在当前池中会自动忽略。
IC_RISK_ASSET_EXCLUSIONS = ["BIL", "SH", "PSQ", "RWM"]

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
    "Momentum_EQ_KAMA": {"color": "#CC79A7", "linewidth": 2.8,
                         "linestyle": "-.", "alpha": 1.0, "zorder": 10},
    "Momentum_EPO_NoKAMA": {"color": "#9467BD", "linewidth": 2.6,
                            "linestyle": ":", "alpha": 1.0, "zorder": 9},
    "Momentum_EQ": {"color": "#E69F00", "linewidth": 3.0,
                    "linestyle": "--", "alpha": 1.0, "zorder": 8},
}

MOMENTUM_WINDOWS = [34]     # 主版本
USE_KAMA_FILTER = True      # KAMA 总开关：控制两条名称含 KAMA 的策略；False 时与无 KAMA 对照重合。
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
EPO_SHRINKAGE = 0.10
DEFENSIVE_ETF = "BIL"        # 无合格候选时持有 BIL；上述开关决定 KAMA 开启时能否正常入选。
MOMENTUM_STRATEGY_ORDER = [
    "Momentum_kama",          # EPO + KAMA
    "Momentum_EQ_KAMA",       # 等权 + KAMA
    "Momentum_EPO_NoKAMA",    # EPO + 无 KAMA
    "Momentum_EQ",            # 等权 + 无 KAMA
]

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAW_DIR = ROOT / "data" / f"raw_{DATA_START}_{DATA_END}"
WINDOW_TAG = "_".join(map(str, MOMENTUM_WINDOWS))
KAMA_TAG = (f"_KAMA_F{'_'.join(map(str, KAMA_FAST))}"
            f"_S{'_'.join(map(str, KAMA_SLOW))}_D{KAMA_DIFF_DAYS}"
            f"_BIL{int(KAMA_FILTER_DEFENSIVE)}" if USE_KAMA_FILTER else "")
OUT_DIR = ROOT / (("results_expanded" if USE_EXPANDED else "results_core")
                  + f"_v6_signal_diagnostics_mom_{WINDOW_TAG}" + KAMA_TAG)
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
    weights, details = epo_from_moments(s, cov, EPO_SHRINKAGE)
    print("Illustration only: A/B returns=8%/4%, vol=20%/10%, correlation=0.5; "
          f"shrinkage={EPO_SHRINKAGE:g}")
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
    priority = {"Momentum_kama": 0, "Momentum_EQ_KAMA": 1,
                "Momentum_EPO_NoKAMA": 2, "Momentum_EQ": 3,
                "SPY": 4, "QQQ": 5, "EqualWeight": 6}
    order = sorted(range(len(labels)), key=lambda i: priority.get(labels[i], 5))
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                       ncol=ncol, fontsize=fontsize, handlelength=3.5)
    for label in legend.get_texts():
        if label.get_text() in STRATEGY_PLOT_STYLES:
            label.set_fontweight("bold")


def summarize(name, ledger):
    """计算单条净值曲线的年化表现；Sortino 只惩罚低于现金收益的波动。"""
    r = ledger["return"]
    excess = r - ledger.rf
    years = (ledger.index[-1] - ledger.start_open.iloc[0]).days / 365.25
    nav = np.r_[1.0, ledger.NAV.to_numpy()]
    max_dd = (nav / np.maximum.accumulate(nav) - 1).min()
    sd = excess.std(ddof=1)
    cagr = ledger.NAV.iloc[-1] ** (1 / years) - 1
    downside = np.minimum(excess.to_numpy(float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(252))
    sortino = (float(excess.mean()) * 252 / downside_deviation
               if downside_deviation > 1e-12 else np.nan)
    calmar = cagr / abs(max_dd) if max_dd < -1e-12 else np.nan
    return {"Strategy": name, "CAGR": cagr,
            "Volatility": r.std(ddof=1) * np.sqrt(252),
            "Sharpe": excess.mean() / sd * np.sqrt(252) if sd > 1e-12 else np.nan,
            "Sortino": sortino, "Calmar": calmar,
            "MaxDrawdown": max_dd, "MeanExposure": ledger.gross.mean(),
            "AnnualTurnover": ledger.turnover.iloc[1:].sum() / years}


def add_information_ratios(performance, results, benchmark="EqualWeight"):
    """加入相对指定基准的年化 Information Ratio。

    基准使用代码中实际回测的月度再平衡 EqualWeight 组合，而不是归因模块的
    日频简单等权收益。所有序列按相同开盘持有区间对齐。
    """
    if benchmark not in results:
        raise ValueError(f"Information-ratio benchmark missing: {benchmark}")
    benchmark_return = results[benchmark]["return"]
    values = {}
    for name, ledger in results.items():
        aligned = pd.concat(
            [ledger["return"].rename("strategy"), benchmark_return.rename("benchmark")],
            axis=1, join="inner").dropna()
        active = aligned.strategy - aligned.benchmark
        tracking_error = active.std(ddof=1)
        values[name] = (active.mean() / tracking_error * np.sqrt(252)
                        if len(active) > 1 and tracking_error > 1e-12 else np.nan)
    performance["InformationRatio_vs_EqualWeight"] = pd.Series(values)
    return performance


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


# ==================== 7. Signal diagnostics and key metrics ====================
def _pearson_and_rank_ic(signal, forward_return):
    """在同一资产横截面计算 Pearson IC 和 Spearman RankIC。"""
    aligned = pd.concat(
        [pd.Series(signal, name="signal"),
         pd.Series(forward_return, name="forward_return")], axis=1).dropna()
    aligned = aligned[np.isfinite(aligned).all(axis=1)]
    if len(aligned) < 3:
        return np.nan, np.nan, len(aligned)
    ic = aligned.signal.corr(aligned.forward_return)
    rank_ic = aligned.signal.rank(method="average").corr(
        aligned.forward_return.rank(method="average"))
    return float(ic), float(rank_ic), len(aligned)


def _momentum_without_r2(scores, windows):
    """构造不乘 R² 的动量信号；多窗口时仍按各窗口名次等权合成。"""
    trend = pd.DataFrame({
        n: pd.to_numeric(scores[f"annualized_trend_{n}"], errors="coerce")
        for n in windows
    })
    valid = trend.notna().all(axis=1)
    if len(windows) == 1:
        return trend.iloc[:, 0].where(valid)
    ranks = trend.where(valid, np.nan, axis=0).rank(
        ascending=False, method="average")
    # 数值越大代表动量越强；负平均名次与当前“名次越小越好”完全等价。
    return (-ranks.mean(axis=1, skipna=False)).where(valid)


def export_signal_diagnostics(close, adj_open, kama_cache, momentum_cache):
    """输出无前视的月频信号诊断。

    信号使用调仓前一交易日收盘数据；未来收益从本次调仓开盘计算到
    下一次调仓开盘。最后一个没有完整下一调仓期的信号自动排除。
    """
    windows, required = momentum_settings()
    score_by_date = {
        pd.Timestamp(key[0]): value.copy()
        for key, value in momentum_cache.items()
        if tuple(key[1]) == tuple(windows) and key[2] == required
    }
    signal_dates = sorted(score_by_date)
    if len(signal_dates) < 2:
        raise ValueError("Signal diagnostics need at least two monthly rebalance signals")

    risk_assets = [t for t in TICKERS if t not in set(IC_RISK_ASSET_EXCLUSIONS)]
    universes = {"All_ETFs": list(TICKERS),
                 "RiskAssets_Ex_BIL_Inverse": risk_assets}
    if len(risk_assets) < 3:
        raise ValueError("Risk-asset IC universe must contain at least three ETFs")

    ic_rows, group_rows, kama_rows = [], [], []
    for signal_date, next_signal_date in zip(signal_dates[:-1], signal_dates[1:]):
        signal_position = close.index.get_loc(signal_date)
        next_signal_position = close.index.get_loc(next_signal_date)
        execution_position = signal_position + 1
        next_execution_position = next_signal_position + 1
        if next_execution_position >= len(adj_open):
            continue
        execution_open = adj_open.index[execution_position]
        next_execution_open = adj_open.index[next_execution_position]
        forward_return = (adj_open.iloc[next_execution_position]
                          / adj_open.iloc[execution_position] - 1)

        scores = score_by_date[signal_date].reindex(TICKERS)
        with_r2 = pd.to_numeric(scores["score"], errors="coerce")
        without_r2 = _momentum_without_r2(scores, windows)
        definitions = {"Momentum_x_R2": with_r2,
                       "Momentum_without_R2": without_r2}

        for universe_name, members in universes.items():
            future = forward_return.reindex(members)
            for definition, signal in definitions.items():
                ic, rank_ic, asset_count = _pearson_and_rank_ic(
                    signal.reindex(members), future)
                ic_rows.append({
                    "SignalClose": signal_date,
                    "ExecutionOpen": execution_open,
                    "NextExecutionOpen": next_execution_open,
                    "Universe": universe_name,
                    "SignalDefinition": definition,
                    "AssetCount": asset_count,
                    "IC": ic,
                    "RankIC": rank_ic,
                })

            # 分组收益只检验策略实际采用的“动量 × R²”排序。
            panel = pd.concat(
                [with_r2.reindex(members).rename("signal"),
                 future.rename("forward_return")], axis=1).dropna()
            panel = panel[np.isfinite(panel).all(axis=1)].sort_values(
                "signal", ascending=False, kind="stable")
            if len(panel) >= SIGNAL_GROUP_COUNT:
                groups = np.array_split(panel.index.to_numpy(), SIGNAL_GROUP_COUNT)
                returns = [float(panel.loc[group, "forward_return"].mean())
                           for group in groups]
                group_rows.append({
                    "SignalClose": signal_date,
                    "ExecutionOpen": execution_open,
                    "NextExecutionOpen": next_execution_open,
                    "Universe": universe_name,
                    "HighReturn": returns[0],
                    "MiddleReturn": returns[1],
                    "LowReturn": returns[2],
                    "HighMinusLow": returns[0] - returns[2],
                    "Monotonic": returns[0] > returns[1] > returns[2],
                    "AssetCount": len(panel),
                })

        # KAMA 是二元过滤器，因此比较通过组和未通过组的下一期等权收益。
        kama_rows_at_date = []
        for ticker in TICKERS:
            row = kama_cache[ticker].loc[signal_date]
            available = (pd.notna(row.kama_fast) and pd.notna(row.kama_slow)
                         and pd.notna(row.kama_fast_diff)
                         and pd.notna(forward_return.get(ticker, np.nan)))
            if available:
                kama_rows_at_date.append({
                    "Ticker": ticker,
                    "Pass": bool(row.kama_eligible),
                    "ForwardReturn": float(forward_return[ticker]),
                })
        kama_panel = pd.DataFrame(kama_rows_at_date)
        if not kama_panel.empty:
            passed = kama_panel.loc[kama_panel.Pass, "ForwardReturn"]
            failed = kama_panel.loc[~kama_panel.Pass, "ForwardReturn"]
            pass_return = passed.mean() if len(passed) else np.nan
            fail_return = failed.mean() if len(failed) else np.nan
            kama_rows.append({
                "SignalClose": signal_date,
                "ExecutionOpen": execution_open,
                "NextExecutionOpen": next_execution_open,
                "PassCount": len(passed),
                "FailCount": len(failed),
                "KAMAPassReturn": pass_return,
                "KAMAFailReturn": fail_return,
                "KAMAPassMinusFail": pass_return - fail_return,
            })

    ic_time = pd.DataFrame(ic_rows)
    group_time = pd.DataFrame(group_rows)
    kama_time = pd.DataFrame(kama_rows)
    if ic_time.empty or group_time.empty or kama_time.empty:
        raise ValueError("Signal diagnostics produced an empty output")

    ic_summary_rows = []
    for (universe, definition), part in ic_time.groupby(
            ["Universe", "SignalDefinition"], sort=False):
        ic_std = part.IC.std(ddof=1)
        rank_std = part.RankIC.std(ddof=1)
        ic_summary_rows.append({
            "Universe": universe,
            "SignalDefinition": definition,
            "MeanIC": part.IC.mean(),
            "MeanRankIC": part.RankIC.mean(),
            "ICStd": ic_std,
            "RankICStd": rank_std,
            "ICIR": part.IC.mean() / ic_std if ic_std > 1e-12 else np.nan,
            "RankICIR": (part.RankIC.mean() / rank_std
                           if rank_std > 1e-12 else np.nan),
            "Observations": int(part.RankIC.count()),
        })
    ic_summary = pd.DataFrame(ic_summary_rows)

    group_summary_rows = []
    for universe, part in group_time.groupby("Universe", sort=False):
        group_summary_rows.append({
            "Universe": universe,
            "AverageHighReturn": part.HighReturn.mean(),
            "AverageMiddleReturn": part.MiddleReturn.mean(),
            "AverageLowReturn": part.LowReturn.mean(),
            "AverageHighMinusLow": part.HighMinusLow.mean(),
            "MonotonicityRate": part.Monotonic.mean(),
            "Observations": len(part),
        })
    group_summary = pd.DataFrame(group_summary_rows)

    kama_summary = pd.DataFrame([{
        "AverageKAMAPassReturn": kama_time.KAMAPassReturn.mean(),
        "AverageKAMAFailReturn": kama_time.KAMAFailReturn.mean(),
        "AverageKAMAPassMinusFail": kama_time.KAMAPassMinusFail.mean(),
        "Observations": int(kama_time.KAMAPassMinusFail.count()),
    }])

    r2_rows = []
    for universe in universes:
        part = ic_summary.set_index(["Universe", "SignalDefinition"])
        with_row = part.loc[(universe, "Momentum_x_R2")]
        without_row = part.loc[(universe, "Momentum_without_R2")]
        r2_rows.append({
            "Universe": universe,
            "MeanIC_WithR2": with_row.MeanIC,
            "MeanIC_WithoutR2": without_row.MeanIC,
            "MeanIC_Difference": with_row.MeanIC - without_row.MeanIC,
            "MeanRankIC_WithR2": with_row.MeanRankIC,
            "MeanRankIC_WithoutR2": without_row.MeanRankIC,
            "MeanRankIC_Difference": with_row.MeanRankIC - without_row.MeanRankIC,
            "ICIR_WithR2": with_row.ICIR,
            "ICIR_WithoutR2": without_row.ICIR,
            "RankICIR_WithR2": with_row.RankICIR,
            "RankICIR_WithoutR2": without_row.RankICIR,
        })
    r2_comparison = pd.DataFrame(r2_rows)

    outputs = {
        "ic_time_series": ic_time,
        "ic_summary": ic_summary,
        "group_time_series": group_time,
        "group_summary": group_summary,
        "r2_comparison": r2_comparison,
        "kama_time_series": kama_time,
        "kama_summary": kama_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT_DIR / f"signal_{name}.csv", index=False,
                     encoding="utf-8-sig")
    return outputs


def completed_position_trades(ledger, audit):
    """把逐资产持仓还原为已完成交易；每只 ETF 从建仓到清仓算一笔。

    月度调仓时只要仍持有该 ETF，就延续同一笔交易。每个持有区间的资产
    盈亏和分摊交易费都计入这笔交易；清仓当天的卖出费用也要计入。
    回测结束时尚未清仓的仓位没有确定的退出时间，因此不计入胜率、
    平均持仓时间和平均每笔利润，但会另报未平仓数。
    """
    holdings = audit["holding_values"]
    pnl = audit["asset_pnl"]
    fees = audit["asset_fees"]
    if not all(frame.index.equals(ledger.index) for frame in (holdings, pnl, fees)):
        raise ValueError("交易指标所用的持仓、盈亏和费用日期未对齐")

    active, completed = {}, []
    for i, row in enumerate(ledger.itertuples()):
        opening_equity = float(row.equity_start)
        # 忽略浮点残差；阈值是当期组合净值的十亿分之一。
        held_now = holdings.iloc[i].gt(opening_equity * 1e-9)
        for ticker in holdings.columns:
            fee = float(fees.iloc[i][ticker])
            if ticker in active and not held_now[ticker]:
                # 本期开盘已清仓：仅将本期开盘的卖出费计入原交易；
                # 本区间不再持有该资产，因此没有后续价格盈亏。
                position = active.pop(ticker)
                position["NetPnL_NAVUnits"] -= fee
                position["HoldingCalendarDays"] = (row.start_open - position["EntryOpen"]).days
                completed.append(position)
            elif held_now[ticker]:
                if ticker not in active:
                    active[ticker] = {"Ticker": ticker, "EntryOpen": row.start_open,
                                      "NetPnL_NAVUnits": 0.0}
                # 包括建仓、持有和部分调仓期间的价格盈亏及交易费。
                active[ticker]["NetPnL_NAVUnits"] += float(pnl.iloc[i][ticker]) - fee

    return pd.DataFrame(completed), len(active)


def export_key_metrics(results, audits, performance, diagnostics):
    """输出组合关键指标，并把所选信号检验汇总进简要 Markdown 报告。"""
    rows = []
    for name in MOMENTUM_STRATEGY_ORDER:
        ledger = results[name]
        trades, open_count = completed_position_trades(ledger, audits[name])
        # 回撤段沿用现有 drawdown_episodes 定义：从净值跌破新高
        # 到恢复新高为一次；期末尚未恢复的回撤也算一次。
        episodes = drawdown_episodes(ledger)
        rows.append({
            "Strategy": name,
            "AnnualReturn": performance.loc[name, "CAGR"],
            "SharpeRatio": performance.loc[name, "Sharpe"],
            "SortinoRatio": performance.loc[name, "Sortino"],
            "CalmarRatio": performance.loc[name, "Calmar"],
            "InformationRatio_vs_EqualWeight": performance.loc[
                name, "InformationRatio_vs_EqualWeight"],
            "MaxDrawdown": performance.loc[name, "MaxDrawdown"],
            "WinRate": (trades.NetPnL_NAVUnits.gt(0).mean() if not trades.empty else np.nan),
            "AnnualVolatility": performance.loc[name, "Volatility"],
            "DrawdownEpisodes": len(episodes),
            "CompletedTrades": len(trades),
            "AverageHoldingCalendarDays": (trades.HoldingCalendarDays.mean()
                                           if not trades.empty else np.nan),
            # 初始净值固定为 1，故 0.02 表示平均每笔净赚初始资金的 2%。
            "AverageNetProfitPerTrade_InitialNAV": (trades.NetPnL_NAVUnits.mean()
                                                    if not trades.empty else np.nan),
            "OpenPositionsAtEnd": open_count,
        })
    metrics = pd.DataFrame(rows).set_index("Strategy")
    metrics.to_csv(OUT_DIR / "key_metrics.csv", encoding="utf-8-sig")

    # Markdown 版只负责显示；CSV 保留未格式化的数值，便于后续制表或计算。
    display = metrics.reset_index().rename(columns={
        "Strategy": "策略", "AnnualReturn": "年化收益", "SharpeRatio": "夏普比率",
        "SortinoRatio": "Sortino", "CalmarRatio": "Calmar",
        "InformationRatio_vs_EqualWeight": "Information Ratio/等权",
        "MaxDrawdown": "最大回撤", "WinRate": "胜率", "AnnualVolatility": "年化波动率",
        "DrawdownEpisodes": "回撤次数", "CompletedTrades": "已完成交易数",
        "AverageHoldingCalendarDays": "平均持仓天数",
        "AverageNetProfitPerTrade_InitialNAV": "平均每笔净利润/初始资金",
        "OpenPositionsAtEnd": "期末未平仓数",
    })
    for col in ["年化收益", "最大回撤", "胜率", "年化波动率"]:
        display[col] = display[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    for col in ["夏普比率", "Sortino", "Calmar", "Information Ratio/等权"]:
        display[col] = display[col].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    display["平均持仓天数"] = display["平均持仓天数"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    display["平均每笔净利润/初始资金"] = display["平均每笔净利润/初始资金"].map(
        lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    notes = [
        "# 回测关键指标", "", markdown_table(display), "",
        "口径：年化收益、夏普、Sortino、Calmar、最大回撤和年化波动率沿用 performance.csv；收益和回撤均为扣交易费后的组合结果。",
        "Sortino 以现金收益为最低可接受收益，只惩罚负的日超额收益；Calmar＝CAGR／|最大回撤|。",
        "Information Ratio 使用策略日收益减去代码中实际回测的月度再平衡 EqualWeight 日收益，再以 sqrt(252) 年化；不是归因模块的日频等权基准。",
        "交易定义：单只 ETF 从首次买入到完全卖出算一笔；期间增减仓仍属同一笔。期末未平仓仓位单列，不进入已完成交易统计。",
        "胜率＝净利润大于零的已完成交易数／已完成交易数；平均持仓天数按买入开盘至清仓开盘的自然日计算。",
        "平均每笔净利润＝已完成交易的逐资产价格盈亏减去对应买卖交易费后的平均值，除以初始组合净值 1；不含现金利息。",
        "回撤次数＝净值跌破此前最高值后持续低于该峰值的一段；期末尚未恢复的回撤也计入。",
        "CSV 中收益、波动、胜率和平均每笔净利润均为小数，例如 0.05 表示 5%。", "",
    ]
    if diagnostics:
        ic_display = diagnostics["ic_summary"].copy().rename(columns={
            "Universe": "资产口径", "SignalDefinition": "信号",
            "MeanIC": "MeanIC", "MeanRankIC": "MeanRankIC",
            "ICStd": "ICStd", "RankICStd": "RankICStd",
            "ICIR": "ICIR", "RankICIR": "RankICIR", "Observations": "期数",
        })
        ic_display["资产口径"] = ic_display["资产口径"].replace({
            "All_ETFs": "全部ETF",
            "RiskAssets_Ex_BIL_Inverse": "剔除BIL与反向ETF",
        })
        ic_display["信号"] = ic_display["信号"].replace({
            "Momentum_x_R2": "动量×R²",
            "Momentum_without_R2": "动量（不乘R²）",
        })
        for col in ["MeanIC", "MeanRankIC", "ICStd", "RankICStd", "ICIR", "RankICIR"]:
            ic_display[col] = ic_display[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—")

        group_display = diagnostics["group_summary"].copy().rename(columns={
            "Universe": "资产口径", "AverageHighReturn": "高分组平均收益",
            "AverageMiddleReturn": "中分组平均收益", "AverageLowReturn": "低分组平均收益",
            "AverageHighMinusLow": "高减低", "MonotonicityRate": "单调月份比例",
            "Observations": "期数",
        })
        group_display["资产口径"] = group_display["资产口径"].replace({
            "All_ETFs": "全部ETF",
            "RiskAssets_Ex_BIL_Inverse": "剔除BIL与反向ETF",
        })
        for col in ["高分组平均收益", "中分组平均收益", "低分组平均收益", "高减低", "单调月份比例"]:
            group_display[col] = group_display[col].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—")

        r2_display = diagnostics["r2_comparison"].copy().rename(columns={
            "Universe": "资产口径", "MeanRankIC_WithR2": "乘R² MeanRankIC",
            "MeanRankIC_WithoutR2": "不乘R² MeanRankIC",
            "MeanRankIC_Difference": "RankIC差值",
            "RankICIR_WithR2": "乘R² RankICIR",
            "RankICIR_WithoutR2": "不乘R² RankICIR",
        })[["资产口径", "乘R² MeanRankIC", "不乘R² MeanRankIC", "RankIC差值",
            "乘R² RankICIR", "不乘R² RankICIR"]]
        r2_display["资产口径"] = r2_display["资产口径"].replace({
            "All_ETFs": "全部ETF",
            "RiskAssets_Ex_BIL_Inverse": "剔除BIL与反向ETF",
        })
        for col in r2_display.columns[1:]:
            r2_display[col] = r2_display[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—")

        kama_display = diagnostics["kama_summary"].copy().rename(columns={
            "AverageKAMAPassReturn": "KAMA通过组平均收益",
            "AverageKAMAFailReturn": "KAMA未通过组平均收益",
            "AverageKAMAPassMinusFail": "通过减未通过",
            "Observations": "期数",
        })
        for col in ["KAMA通过组平均收益", "KAMA未通过组平均收益", "通过减未通过"]:
            kama_display[col] = kama_display[col].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—")

        notes += [
            "## IC 与 RankIC", "", markdown_table(ic_display), "",
            "每期在 ETF 横截面计算：信号来自调仓前一日收盘，未来收益为本次调仓开盘至下次调仓开盘。ICIR 和 RankICIR 均为月度均值除以月度标准差，未额外年化。", "",
            "## 三组收益与单调性", "", markdown_table(group_display), "",
            "按动量×R²从高到低尽量等分为 High/Middle/Low 三组；单调月份要求 High > Middle > Low。", "",
            "## R² 有效性对照", "", markdown_table(r2_display), "",
            "RankIC差值＝乘R²版本减去不乘R²版本；正值表示R²改善了平均横截面排序能力。", "",
            "## KAMA 过滤检验", "", markdown_table(kama_display), "",
            "KAMA通过减未通过＝同期通过KAMA资产的下一调仓期等权平均收益－未通过资产的平均收益。", "",
        ]
    (OUT_DIR / "key_metrics.md").write_text("\n".join(notes), encoding="utf-8")
    return metrics


def plot_key_metrics(metrics, diagnostics, ledger):
    """将旧的单表图片替换为组合表现与信号检验完整版仪表板。"""
    headers = ["Strategy", "Annual\nReturn", "Sharpe", "Sortino", "Calmar",
               "Max\nDrawdown", "Win\nRate", "Annual\nVolatility",
               "IR vs EW", "Completed\nTrades", "Avg. Holding\nDays",
               "Avg. Net Profit\nper Trade*"]
    cells = []
    for name, row in metrics.iterrows():
        cells.append([
            name, f"{row.AnnualReturn:.2%}", f"{row.SharpeRatio:.3f}",
            f"{row.SortinoRatio:.3f}", f"{row.CalmarRatio:.3f}",
            f"{row.MaxDrawdown:.2%}",
            f"{row.WinRate:.2%}" if pd.notna(row.WinRate) else "—",
            f"{row.AnnualVolatility:.2%}",
            f"{row.InformationRatio_vs_EqualWeight:.3f}"
            if pd.notna(row.InformationRatio_vs_EqualWeight) else "—",
            str(int(row.CompletedTrades)),
            f"{row.AverageHoldingCalendarDays:.1f}"
            if pd.notna(row.AverageHoldingCalendarDays) else "—",
            f"{row.AverageNetProfitPerTrade_InitialNAV:+.2%}"
            if pd.notna(row.AverageNetProfitPerTrade_InitialNAV) else "—",
        ])

    if diagnostics:
        fig = plt.figure(figsize=(22, 18), facecolor="white")
        grid = fig.add_gridspec(4, 2, height_ratios=[1.25, 1, 1, 1],
                                hspace=.42, wspace=.22)
        table_ax = fig.add_subplot(grid[0, :])
    else:
        fig = plt.figure(figsize=(22, 5.5), facecolor="white")
        grid = fig.add_gridspec(1, 1)
        table_ax = fig.add_subplot(grid[0, 0])
    table_ax.axis("off")
    table_ax.text(0.5, 1.03, "Performance and Signal Diagnostics | Four Momentum Strategies",
                  ha="center", va="center", fontsize=20, fontweight="bold",
                  color="#17324F", transform=table_ax.transAxes)
    period = (f"Backtest period: {ledger.start_open.iloc[0].date()} "
              f"to {ledger.index[-1].date()}")
    table_ax.text(0.5, .94, period, ha="center", va="center", fontsize=11,
                  color="#52657A", transform=table_ax.transAxes)
    table = table_ax.table(
        cellText=cells, colLabels=headers, cellLoc="center",
        colWidths=[.16, .07, .06, .06, .06, .075, .06, .075, .065, .07, .085, .12],
        bbox=[.005, .18, .99, .67])
    table.auto_set_font_size(False)
    for (row_no, col_no), cell in table.get_celld().items():
        cell.set_edgecolor("#D5DFE9")
        cell.set_linewidth(.8)
        cell.get_text().set_fontsize(10 if row_no == 0 else 9.5)
        if row_no == 0:
            cell.set_facecolor("#17324F")
            cell.get_text().set_color("white")
        else:
            cell.set_facecolor("#F0F5FA" if row_no % 2 else "white")
            cell.get_text().set_color("#172B42")
        if col_no == 0 and row_no > 0:
            cell.get_text().set_ha("left")
    table_ax.text(
        .01, .08,
        "* Closed ETF-position trades only; net of allocated trading costs and expressed as a percentage of initial portfolio value.",
        ha="left", va="center", fontsize=10, color="#42566C",
        transform=table_ax.transAxes)

    if diagnostics:
        colors = ["#0072B2", "#E69F00", "#6F4E9C", "#8A8A8A"]
        ic = diagnostics["ic_summary"].copy()
        ic["Label"] = (ic.Universe.map({
            "All_ETFs": "All",
            "RiskAssets_Ex_BIL_Inverse": "Risk",
        }) + " | " + ic.SignalDefinition.map({
            "Momentum_x_R2": "Mom×R²",
            "Momentum_without_R2": "Mom only",
        }))
        x = np.arange(len(ic))

        ax = fig.add_subplot(grid[1, 0])
        width = .36
        ax.bar(x-width/2, ic.MeanIC, width, label="Mean IC", color="#0072B2")
        ax.bar(x+width/2, ic.MeanRankIC, width, label="Mean RankIC",
               color="#E69F00", hatch="//")
        ax.set_xticks(x, ic.Label, rotation=18, ha="right")
        ax.set_title("Cross-sectional IC and RankIC")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
        ax.legend()

        ax = fig.add_subplot(grid[1, 1])
        ax.bar(x-width/2, ic.ICIR, width, label="ICIR", color="#6F4E9C")
        ax.bar(x+width/2, ic.RankICIR, width, label="RankICIR",
               color="#8A8A8A", hatch="xx")
        ax.set_xticks(x, ic.Label, rotation=18, ha="right")
        ax.set_title("Signal stability (mean / time-series standard deviation)")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
        ax.legend()

        groups = diagnostics["group_summary"].copy()
        gx = np.arange(len(groups))
        ax = fig.add_subplot(grid[2, :])
        width = .22
        ax.bar(gx-width, groups.AverageHighReturn, width, label="High score",
               color="#0072B2")
        ax.bar(gx, groups.AverageMiddleReturn, width, label="Middle score",
               color="#E69F00", hatch="//")
        ax.bar(gx+width, groups.AverageLowReturn, width, label="Low score",
               color="#8A8A8A", hatch="xx")
        group_labels = [
            ("All ETFs" if row.Universe == "All_ETFs" else "Risk assets")
            + f"\nMonotonic months: {row.MonotonicityRate:.1%}\nHigh-Low: {row.AverageHighMinusLow:+.2%}"
            for row in groups.itertuples()
        ]
        ax.set_xticks(gx, group_labels)
        ax.set_title("Next-rebalance-period returns by Momentum × R² group")
        ax.set_ylabel("Average forward return")
        ax.yaxis.set_major_formatter(lambda value, position: f"{value:.1%}")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
        ax.legend(ncol=3)

        r2 = diagnostics["r2_comparison"]
        rx = np.arange(len(r2))
        ax = fig.add_subplot(grid[3, 0])
        width = .36
        ax.bar(rx-width/2, r2.MeanRankIC_WithR2, width, label="Momentum × R²",
               color="#0072B2")
        ax.bar(rx+width/2, r2.MeanRankIC_WithoutR2, width,
               label="Momentum only", color="#E69F00", hatch="//")
        ax.set_xticks(rx, ["All ETFs", "Risk assets"])
        ax.set_title("Does R² improve ranking ability?")
        ax.set_ylabel("Mean RankIC")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
        ax.legend()

        kama = diagnostics["kama_summary"].iloc[0]
        ax = fig.add_subplot(grid[3, 1])
        labels = ["KAMA pass", "KAMA fail", "Pass − fail"]
        values = [kama.AverageKAMAPassReturn, kama.AverageKAMAFailReturn,
                  kama.AverageKAMAPassMinusFail]
        bars = ax.bar(labels, values, color=colors[:3])
        bars[1].set_hatch("//")
        bars[2].set_hatch("xx")
        ax.set_title("KAMA filter: next-rebalance-period return")
        ax.set_ylabel("Average forward return")
        ax.yaxis.set_major_formatter(lambda value, position: f"{value:.1%}")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.grid(axis="y", alpha=.2)

    if diagnostics:
        fig.text(
            .01, .008,
            "Signal timing: previous trading-day close → current rebalance open → next rebalance open. "
            "Risk assets exclude BIL, SH, PSQ and RWM. ICIR values are not annualized.",
            ha="left", va="bottom", fontsize=10, color="#42566C")
    fig.savefig(OUT_DIR / "key_metrics.png", dpi=180, facecolor="white",
                bbox_inches="tight")
    return fig


def period_attribution(ledger, audit, selected_rows):
    """同时计算绝对贡献和相对同期全资产等权组合的主动贡献。

    绝对净贡献可精确加总为该期累计收益，用于净值对账。
    相对毛贡献逐持有区间计算为：
        实际持仓金额 ×（该资产收益率 - 当日全资产简单等权收益率）
    相对净贡献再扣除归属于该资产的实际交易费用。现金同样减去等权基准收益。
    该主动 PnL 分解是日频金额差的加总，不等于两条独立净值曲线最终收益率之差。
    """
    part = ledger.loc[selected_rows]
    if part.empty:
        return None, None
    initial, final = float(part.equity_start.iloc[0]), float(part.NAV.iloc[-1])
    asset_returns = audit["asset_returns"].loc[part.index]
    holding_values = audit["holding_values"].loc[part.index]
    asset_pnl = audit["asset_pnl"].loc[part.index]
    gross = asset_pnl.sum()
    fees = audit["asset_fees"].loc[part.index].sum()
    equal_weight_return = asset_returns.mean(axis=1)
    excess_returns = asset_returns.sub(equal_weight_return, axis=0)
    relative_gross = holding_values.mul(excess_returns).sum()
    held = holding_values.abs().gt(1e-12)
    equal_weight_panel = asset_returns.mul(0).add(equal_weight_return, axis=0)

    components = pd.DataFrame({
        "AbsoluteGrossPnL": gross,
        "FeesPaid": fees,
        "RelativeGrossPnL": relative_gross,
        "HeldIntervals": held.sum(),
        "MeanAssetReturnWhenHeld": asset_returns.where(held).mean(),
        "MeanEqualWeightReturnWhenHeld": equal_weight_panel.where(held).mean(),
        "MeanExcessReturnWhenHeld": excess_returns.where(held).mean(),
    })
    post_trade_equity = part.equity_start - part.fee
    cash_value = part.cash_weight * post_trade_equity
    cash_held = cash_value.abs().gt(1e-12)
    cash_return = part.cash_pnl.div(cash_value.where(cash_held))
    cash_relative_gross = (part.cash_pnl - cash_value * equal_weight_return).sum()
    components.loc["CASH_INTEREST"] = {
        "AbsoluteGrossPnL": part.cash_pnl.sum(),
        "FeesPaid": 0.0,
        "RelativeGrossPnL": cash_relative_gross,
        "HeldIntervals": int(cash_held.sum()),
        "MeanAssetReturnWhenHeld": cash_return.where(cash_held).mean(),
        "MeanEqualWeightReturnWhenHeld": equal_weight_return.where(cash_held).mean(),
        "MeanExcessReturnWhenHeld": (
            cash_return - equal_weight_return).where(cash_held).mean(),
    }
    components["AbsoluteNetPnL"] = (
        components.AbsoluteGrossPnL - components.FeesPaid)
    components["RelativeNetPnL"] = components.RelativeGrossPnL - components.FeesPaid
    components["AbsoluteGrossContribution"] = components.AbsoluteGrossPnL / initial
    components["FeeContribution"] = -components.FeesPaid / initial
    components["AbsoluteNetContribution"] = components.AbsoluteNetPnL / initial
    components["RelativeGrossContribution"] = components.RelativeGrossPnL / initial
    components["RelativeNetContribution"] = components.RelativeNetPnL / initial

    absolute_reconstructed = float(components.AbsoluteNetPnL.sum())
    absolute_error = absolute_reconstructed - (final-initial)
    if not np.isclose(absolute_reconstructed, final-initial, atol=1e-10, rtol=1e-9):
        raise AssertionError("Absolute period attribution does not reconcile")
    relative_target = float((asset_pnl.sum(axis=1) + part.cash_pnl - part.fee
                             - post_trade_equity * equal_weight_return).sum())
    relative_reconstructed = float(components.RelativeNetPnL.sum())
    relative_error = relative_reconstructed - relative_target
    if not np.isclose(relative_reconstructed, relative_target, atol=1e-10, rtol=1e-9):
        raise AssertionError("Relative period attribution does not reconcile")
    benchmark_period_return = float((1 + equal_weight_return).prod() - 1)
    summary = {"start_open": part.start_open.iloc[0], "end_open": part.index[-1],
               "start_NAV": initial, "end_NAV": final,
               "period_return": final/initial-1,
               "absolute_contribution_sum": components.AbsoluteNetContribution.sum(),
               "equal_weight_benchmark_return": benchmark_period_return,
               "geometric_return_difference": final/initial-1-benchmark_period_return,
               "relative_contribution_sum": components.RelativeNetContribution.sum(),
               "fees_NAV_units": part.fee.sum(),
               "fees_over_period_start_NAV": part.fee.sum()/initial,
               "absolute_reconciliation_error_NAV_units": absolute_error,
               "relative_reconciliation_error_NAV_units": relative_error,
               "attribution_benchmark": "daily_equal_weight_return_of_all_TICKERS"}
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
    selected_counts = pd.Series(signal.selected_counts, dtype=float)
    holdings = pd.DataFrame({
        "MeanWeight": weights.mean(), "MaxWeight": weights.max(),
        "MeanWeightWhenHeld": weights.where(held).mean(),
        "HeldIntervals": held.sum(), "HeldFraction": held.mean(),
        "SelectedRebalances": selected_counts.reindex(weights.columns).fillna(0).astype(int)})
    full_asset_attribution = tables["Full"].reindex(weights.columns)
    holdings = holdings.join(full_asset_attribution[[
        "AbsoluteGrossPnL", "FeesPaid", "AbsoluteNetPnL",
        "RelativeGrossPnL", "RelativeNetPnL",
        "MeanAssetReturnWhenHeld", "MeanEqualWeightReturnWhenHeld",
        "MeanExcessReturnWhenHeld"]])
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
    """合并四条策略的归因，并输出 2×2 因子差异。"""
    full_rows, period_rows, holding_rows, summary_rows = [], [], [], []
    for name, report in reports.items():
        for period, table in report["tables"].items():
            rows = table.reset_index().assign(Strategy=name, Period=period)
            period_rows.append(rows)
            if period == "Full":
                full_rows.append(rows.drop(columns="Period"))
        holding_rows.append(report["holdings"].reset_index().assign(Strategy=name))
        summary_rows.append(report["summary"].assign(Strategy=name))
    full = pd.concat(full_rows, ignore_index=True).sort_values(
        ["Strategy", "RelativeNetContribution"], ascending=[True, False])
    full.to_csv(OUT_DIR / "asset_contribution_full.csv", index=False, encoding="utf-8-sig")
    pd.concat(period_rows, ignore_index=True).to_csv(
        OUT_DIR / "contribution_by_period.csv", index=False, encoding="utf-8-sig")
    pd.concat(holding_rows, ignore_index=True).to_csv(
        OUT_DIR / "holding_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(summary_rows, ignore_index=True).to_csv(
        OUT_DIR / "attribution_period_summary.csv", index=False, encoding="utf-8-sig")
    strategy_order = [name for name in MOMENTUM_STRATEGY_ORDER if name in reports]
    common_periods = set.intersection(
        *(set(reports[name]["tables"]) for name in strategy_order))
    comparisons = []
    for period in sorted(common_periods):
        # 每条策略自身最大回撤段的日期不同，不能横向比较。
        if period == "Worst_drawdown_peak_to_trough":
            continue
        frame = pd.DataFrame({
            name: reports[name]["tables"][period].RelativeNetContribution
            for name in strategy_order})
        if set(MOMENTUM_STRATEGY_ORDER).issubset(frame.columns):
            frame["EPO_Effect_NoKAMA"] = (
                frame.Momentum_EPO_NoKAMA - frame.Momentum_EQ)
            frame["EPO_Effect_WithKAMA"] = (
                frame.Momentum_kama - frame.Momentum_EQ_KAMA)
            frame["KAMA_Effect_EqualWeight"] = (
                frame.Momentum_EQ_KAMA - frame.Momentum_EQ)
            frame["KAMA_Effect_EPO"] = (
                frame.Momentum_kama - frame.Momentum_EPO_NoKAMA)
        frame.index.name = "Component"
        comparisons.append(frame.reset_index().assign(Period=period))
    comparison = pd.concat(comparisons, ignore_index=True).sort_values(
        ["Period", "Momentum_kama"], ascending=[True, False])
    comparison.to_csv(OUT_DIR / "four_way_relative_contribution.csv",
                      index=False, encoding="utf-8-sig")

    # 全期图的主口径为相对同期等权组合的净贡献，同时使用颜色和纹理区分策略。
    frame = pd.DataFrame({
        name: reports[name]["tables"]["Full"].RelativeNetContribution
        for name in strategy_order}) * 100
    frame = frame.loc[frame.Momentum_kama.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [STRATEGY_PLOT_STYLES[name]["color"] for name in strategy_order]
    frame.plot.bar(ax=ax, color=colors, width=.82)
    hatches = ["", "//", "..", "xx"]
    for container, hatch in zip(ax.containers, hatches):
        for patch in container:
            patch.set_hatch(hatch)
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set(title="Full-period contribution relative to daily equal-weight universe",
           xlabel="", ylabel="Relative net contribution (percentage points of initial NAV)")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "contribution_comparison.png", dpi=160)
    plt.close(fig)
    return comparison


def export_cost_analysis(results, zero_cost_results):
    """费用影响只留一张表；零费用收益来自真正重跑，保留利息假设。"""
    rows = []
    for name in MOMENTUM_STRATEGY_ORDER:
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


def write_report(performance, yearly, reports, costs, results, diagnostics):
    """中文摘要：组合绩效、信号检验、归因、集中度和费用。"""
    fmt_pct = lambda x: f"{x:.2%}" if pd.notna(x) else "—"
    fmt_pp = lambda x: f"{100*x:+.2f}" if pd.notna(x) else "—"
    preferred = MOMENTUM_STRATEGY_ORDER + [
        "SPY", "QQQ", "EqualWeight", "LeveragedEqualWeight", "BIL"]
    selected = [n for n in preferred if n in performance.index]
    perf = performance.loc[selected, [
        "CAGR", "Volatility", "Sharpe", "Sortino", "Calmar",
        "InformationRatio_vs_EqualWeight", "MaxDrawdown", "AnnualTurnover"]].copy()
    for col in ["CAGR", "Volatility", "MaxDrawdown"]:
        perf[col] = perf[col].map(fmt_pct)
    for col in ["Sharpe", "Sortino", "Calmar", "InformationRatio_vs_EqualWeight"]:
        perf[col] = perf[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    perf["AnnualTurnover"] = perf.AnnualTurnover.map(lambda x: f"{x:.2f} 倍")
    perf = perf.rename(columns={"CAGR": "年化收益", "Volatility": "年化波动",
                               "Sharpe": "Sharpe", "Sortino": "Sortino",
                               "Calmar": "Calmar",
                               "InformationRatio_vs_EqualWeight": "Information Ratio/等权",
                               "MaxDrawdown": "最大回撤",
                               "AnnualTurnover": "年化买卖交易量"})
    perf.index.name = "策略"
    ledger = results["Momentum_kama"]
    windows, required = momentum_settings()
    lines = ["# QF5210 ETF 策略回测：动量、KAMA 与 EPO", "",
             f"实际覆盖：{ledger.start_open.iloc[0].date()} 开盘至 {ledger.index[-1].date()} 开盘。",
             f"资产池：{', '.join(TICKERS)}。单边交易成本：{COST_BPS:g} bp。",
             f"KAMA 总开关 USE_KAMA_FILTER={USE_KAMA_FILTER}；控制 Momentum_kama 与 Momentum_EQ_KAMA。快线{KAMA_FAST}，慢线{KAMA_SLOW}，快线{KAMA_DIFF_DAYS}日变化>0；BIL正常入选需通过筛选：{KAMA_FILTER_DEFENSIVE}。",
             f"动量窗口：{windows} 个收盘价格；最多持有 {TOP_N} 只；每月首个交易日开盘调仓，信号使用上一交易日收盘。",
             f"窗口内分数：expm1({MOMENTUM_ANNUALIZATION} × log-price 回归斜率) × R²。",
             f"单窗口按原分数排序；多窗口分别排名后取平均名次，越小越优。要求至少 {required}/{len(windows)} 个窗口分数为正，且所有窗口均有有效数据。",
             "多窗口 score=1/平均名次，仅作排序；筛选看 eligible，不是看合成 score 是否大于零。",
             f"EPO：{EPO_PRICE_DAYS} 个价格估计日收益均值/协方差，收缩参数 {EPO_SHRINKAGE:g}，非负截断后归一化。",
             ("资产池不含 TLT、TIP；固定收益 ETF 仅保留 BIL。无合格候选时满仓 BIL；"
              + ("正常入选时 BIL 必须通过动量与 KAMA。" if USE_KAMA_FILTER and KAMA_FILTER_DEFENSIVE
                 else "KAMA 开启时 BIL 不正常入选，仅作回退。" if USE_KAMA_FILTER
                 else "KAMA 关闭时 BIL 与其他 ETF 一样按动量筛选。")),
             "四条动量策略均使用同一组动量窗口、目标 ETF 权重合计 100%，不使用账户杠杆、目标波动率或周度仓位控制。",
             "2×2 对照：Momentum_EQ=等权+无KAMA；Momentum_EQ_KAMA=等权+KAMA；Momentum_EPO_NoKAMA=EPO+无KAMA；Momentum_kama=EPO+KAMA。",
             "因此 EPO 效果应在相同 KAMA 状态下比较，KAMA 效果应在相同配权方式下比较；避免把两项变化混为一谈。",
             (f"杠杆等权基准：启用={RUN_LEVERAGED_EW}，杠杆上限={MAX_LEVERAGE:g}；只买非 BIL 资产，借款利率为现金利率加融资利差。"
              if RUN_LEVERAGED_EW else "杠杆等权基准未运行；MAX_LEVERAGE 不影响上述四条动量策略。"),
             "删去资产与窗口选择均已使用历史表现作开发判断，未来结果仍需固定规则向前评估。", "",
             "## 绩效", "", markdown_table(perf.reset_index()), "",
             "Sortino 只惩罚低于现金收益的日波动；Calmar＝CAGR／|最大回撤|；Information Ratio 的基准是代码中实际回测的月度再平衡 EqualWeight 组合。",
             "年化买卖交易量是买入金额与卖出金额分别相对交易前净值的总和，排除首次建仓；不是除以二的单边换手。", ""]
    if diagnostics:
        ic = diagnostics["ic_summary"].copy()
        ic["Universe"] = ic.Universe.replace({
            "All_ETFs": "全部ETF",
            "RiskAssets_Ex_BIL_Inverse": "剔除BIL与反向ETF",
        })
        ic["SignalDefinition"] = ic.SignalDefinition.replace({
            "Momentum_x_R2": "动量×R²",
            "Momentum_without_R2": "动量（不乘R²）",
        })
        ic = ic.rename(columns={"Universe": "资产口径", "SignalDefinition": "信号",
                                "Observations": "期数"})
        for col in ["MeanIC", "MeanRankIC", "ICStd", "RankICStd", "ICIR", "RankICIR"]:
            ic[col] = ic[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

        grouped = diagnostics["group_summary"].copy().rename(columns={
            "Universe": "资产口径", "AverageHighReturn": "高分组",
            "AverageMiddleReturn": "中分组", "AverageLowReturn": "低分组",
            "AverageHighMinusLow": "高减低", "MonotonicityRate": "单调月份比例",
            "Observations": "期数",
        })
        grouped["资产口径"] = grouped["资产口径"].replace({
            "All_ETFs": "全部ETF",
            "RiskAssets_Ex_BIL_Inverse": "剔除BIL与反向ETF",
        })
        for col in ["高分组", "中分组", "低分组", "高减低", "单调月份比例"]:
            grouped[col] = grouped[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")

        r2 = diagnostics["r2_comparison"].copy().rename(columns={
            "Universe": "资产口径", "MeanRankIC_WithR2": "乘R² MeanRankIC",
            "MeanRankIC_WithoutR2": "不乘R² MeanRankIC",
            "MeanRankIC_Difference": "差值",
        })[["资产口径", "乘R² MeanRankIC", "不乘R² MeanRankIC", "差值"]]
        r2["资产口径"] = r2["资产口径"].replace({
            "All_ETFs": "全部ETF",
            "RiskAssets_Ex_BIL_Inverse": "剔除BIL与反向ETF",
        })
        for col in r2.columns[1:]:
            r2[col] = r2[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

        kama = diagnostics["kama_summary"].copy().rename(columns={
            "AverageKAMAPassReturn": "KAMA通过组",
            "AverageKAMAFailReturn": "KAMA未通过组",
            "AverageKAMAPassMinusFail": "通过减未通过",
            "Observations": "期数",
        })
        for col in ["KAMA通过组", "KAMA未通过组", "通过减未通过"]:
            kama[col] = kama[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")

        lines += [
            "## 信号有效性", "",
            "信号统一使用调仓前一日收盘，未来收益为本次调仓开盘至下次调仓开盘；最后一个不完整持有期不纳入。", "",
            markdown_table(ic), "",
            "ICIR与RankICIR＝时间序列均值／标准差，未额外年化。", "",
            "### 分组收益与单调性", "", markdown_table(grouped), "",
            "按动量×R²由高到低尽量等分为三组；单调月份要求高分组收益>中分组收益>低分组收益。", "",
            "### R² 是否改善排序", "", markdown_table(r2), "",
            "差值为乘R²版本减去不乘R²版本；正值表示R²提高平均RankIC。", "",
            "### KAMA 过滤", "", markdown_table(kama), "",
            "通过减未通过为两组下一调仓期等权平均收益之差。", "",
        ]
    lines += ["## 年度收益", ""]
    annual = yearly[[n for n in MOMENTUM_STRATEGY_ORDER + ["SPY", "EqualWeight"]
                     if n in yearly]].copy()
    for col in annual:
        annual[col] = annual[col].map(fmt_pct)
    annual.index.name = "年份"
    lines += [markdown_table(annual.reset_index()), "", "首尾年份可能不完整；各年收益按持有区间结束日期归属。", ""]
    if reports:
        epo_kama = reports["Momentum_kama"]
        full = pd.DataFrame({
            "EPO+KAMA": reports["Momentum_kama"]["tables"]["Full"].RelativeNetContribution,
            "等权+KAMA": reports["Momentum_EQ_KAMA"]["tables"]["Full"].RelativeNetContribution,
            "EPO+无KAMA": reports["Momentum_EPO_NoKAMA"]["tables"]["Full"].RelativeNetContribution,
            "等权+无KAMA": reports["Momentum_EQ"]["tables"]["Full"].RelativeNetContribution,
        })
        full = full.sort_values("EPO+KAMA", ascending=False)
        full.loc["合计"] = full.sum()
        full = full.map(fmt_pp)
        full.index.name = "资产"
        lines += ["## 全期相对收益归因", "", markdown_table(full.reset_index()), "",
                  "单位：相对贡献百分点。每个持有区间的相对毛贡献 = 实际持仓金额 ×（资产收益率 − 当日全资产简单等权收益率），相对净贡献再扣除该资产交易费。",
                  "这让 BIL、GLD 等防御资产在全市场下跌时可以记录正的保护贡献；它们不再因为绝对收益低而天然吃亏。相对贡献是日频主动 PnL 的金额加总，不是两条独立净值曲线最终收益率之差。",
                  "asset_contribution_full.csv 同时保留 Absolute* 列用于净值对账；AbsoluteNetContribution 的合计严格等于组合累计收益。", ""]
        summaries = {n: r["summary"].set_index("Period") for n, r in reports.items()}
        stage_rows = []
        for period in ATTRIBUTION_PERIODS:
            if period not in summaries["Momentum_kama"].index:
                continue
            a = summaries["Momentum_kama"].loc[period]
            stage_rows.append({"阶段": period,
                               "实际开始开盘": pd.Timestamp(a.start_open).date(),
                               "实际结束开盘": pd.Timestamp(a.end_open).date(),
                               "EPO+KAMA": fmt_pct(a.period_return),
                               "等权+KAMA": fmt_pct(summaries["Momentum_EQ_KAMA"].loc[period].period_return),
                               "EPO+无KAMA": fmt_pct(summaries["Momentum_EPO_NoKAMA"].loc[period].period_return),
                               "等权+无KAMA": fmt_pct(summaries["Momentum_EQ"].loc[period].period_return),
                               "全资产日等权基准": fmt_pct(a.equal_weight_benchmark_return)})
        lines += ["## 关键阶段", "", markdown_table(pd.DataFrame(stage_rows)), ""]
        episodes = epo_kama["episodes"]
        if not episodes.empty:
            worst = episodes.iloc[0]
            losses = epo_kama["tables"]["Worst_drawdown_peak_to_trough"].RelativeNetContribution.sort_values().head(4)
            lines += [f"Momentum_kama 最大回撤：{worst.drawdown:.2%}；峰值 {worst.peak_date.date()} → 谷底 {worst.trough_date.date()}。",
                      "该段相对净贡献最低的四项：" + "；".join(f"{t} {100*v:+.2f} 个百分点" for t, v in losses.items()) + "。", ""]
        lines += ["## 持仓与集中度", ""]
        risk_rows = []
        for name in MOMENTUM_STRATEGY_ORDER:
            report = reports[name]
            exposure = report["exposure"]
            risk_rows.append({"策略": name, "最大单资产权重": fmt_pct(exposure.MaxAssetWeight.max()),
                              "平均实际持有数": f"{exposure.HeldAssetCount.mean():.2f}",
                              "平均QQQ/XLK/SMH权重": fmt_pct(exposure.Tech_related_ETFs.mean()),
                              "EPO退回等权次数": report["status_counts"].get("fallback_equal", 0)
                              if name in {"Momentum_kama", "Momentum_EPO_NoKAMA"} else "—"})
        lines += [markdown_table(pd.DataFrame(risk_rows)), "",
                  "科技敞口只是 QQQ、XLK、SMH 的 ETF 权重合计，未穿透成分股；入选三只不代表实际持有三只。", ""]
        if "USO" in epo_kama["holdings"].index:
            uso_rows = []
            for name in MOMENTUM_STRATEGY_ORDER:
                holding = reports[name]["holdings"].loc["USO"]
                attr = reports[name]["tables"]["Full"].loc["USO"]
                uso_rows.append({"策略": name,
                                 "相对净贡献(百分点)": fmt_pp(attr.RelativeNetContribution),
                                 "绝对净贡献(百分点)": fmt_pp(attr.AbsoluteNetContribution),
                                 "平均权重": fmt_pct(holding.MeanWeight),
                                 "持有区间占比": fmt_pct(holding.HeldFraction)})
            lines += ["## USO", "", markdown_table(pd.DataFrame(uso_rows)), "",
                      "相对贡献衡量持有 USO 相比同期全资产等权收益的增减益；是否应从资产池删除，仍应另跑固定其他设置的不含 USO 对照。", ""]
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
              "performance.csv：全部 ETF 和组合绩效（含 Sortino、Calmar 和相对等权 Information Ratio）；key_metrics.csv/md/png：四策略关键指标与完整版诊断图。",
              "signal_ic_time_series.csv 与 signal_ic_summary.csv：两种资产口径、乘/不乘R²信号的IC和RankIC；signal_group_time_series.csv 与 signal_group_summary.csv：三组收益和单调性。",
              "signal_r2_comparison.csv：R²前后对照；signal_kama_time_series.csv 与 signal_kama_summary.csv：KAMA通过组减未通过组；nav.csv：全部净值；yearly_returns.csv：年度收益；cost_summary.csv：费用影响。",
              "asset_contribution_full.csv：四策略全期归因；contribution_by_period.csv：全期、关键阶段和各策略自身最大回撤段；holding_summary.csv：持仓；attribution_period_summary.csv：各阶段组合收益与等权基准收益。",
              "four_way_relative_contribution.csv：四策略相对贡献与 EPO/KAMA 的同条件差异；backtest.png：全部净值/回撤；momentum_comparison.png：四策略重点对照；contribution_comparison.png：四策略全期相对贡献。",
              "CSV 的收益、贡献、权重保存为小数，0.05=5%；PnL/FeesPaid 为初始净值=1 的金额单位。Relative* 是相对当日全资产等权收益的主动贡献；Absolute* 是实际绝对贡献并用于净值对账。",
              "各期贡献均除以该期初净值，不能跨期直接相加。RelativeNetContribution 已扣交易费；相对贡献合计是日频主动 PnL 的算术加总，不等于 geometric_return_difference，后者是策略期收益减去独立复利的日等权基准收益。",
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
    if SIGNAL_GROUP_COUNT != 3:
        raise ValueError("SIGNAL_GROUP_COUNT must remain 3 for the High/Middle/Low report")
    if len(set(IC_RISK_ASSET_EXCLUSIONS)) != len(IC_RISK_ASSET_EXCLUSIONS):
        raise ValueError("IC_RISK_ASSET_EXCLUSIONS contains duplicates")
    if DEFENSIVE_ETF not in TICKERS or len(set(TICKERS)) != len(TICKERS):
        raise ValueError("Universe must contain defensive ETF and no duplicates")
    print(f"Universe ({len(TICKERS)}): {TICKERS}")
    print("Four-way comparison: equal/EPO allocation x no-KAMA/KAMA; "
          f"USE_KAMA_FILTER={USE_KAMA_FILTER}; monthly, long-only, target gross=1.0")
    print(f"Momentum windows={windows}; require {required}/{len(windows)} positive windows")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {k: globals()[k] for k in [
        "TICKERS", "USE_EXPANDED", "DATA_START", "DATA_END", "BACKTEST_START",
        "COST_BPS", "CASH_RATE", "BORROW_SPREAD", "MAX_LEVERAGE", "RUN_LEVERAGED_EW",
        "RUN_ATTRIBUTION", "RUN_COST_COUNTERFACTUAL", "ATTRIBUTION_PERIODS",
        "RUN_SIGNAL_DIAGNOSTICS", "SIGNAL_GROUP_COUNT", "IC_RISK_ASSET_EXCLUSIONS",
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
    strategy_specs = [
        ("Momentum_EQ", "equal", False),
        ("Momentum_EQ_KAMA", "equal", USE_KAMA_FILTER),
        ("Momentum_EPO_NoKAMA", "epo", False),
        ("Momentum_kama", "epo", USE_KAMA_FILTER),
    ]
    for name, method, use_kama in strategy_specs:
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
    if RUN_LEVERAGED_EW:
        levered = leveraged_equal_weight_target(TICKERS, MAX_LEVERAGE)
        print(f"LeveragedEqualWeight: {MAX_LEVERAGE:.2f}x across "
              f"{len(levered)} non-{DEFENSIVE_ETF} ETFs; {DEFENSIVE_ETF} weight=0")
        results["LeveragedEqualWeight"] = backtest(
            close, adj_open, "LeveragedEqualWeight", lambda date, history: levered,
            gross_cap=MAX_LEVERAGE)
    performance = pd.DataFrame(
        [summarize(n, x) for n, x in results.items()]).set_index("Strategy")
    performance = add_information_ratios(performance, results)
    performance.to_csv(OUT_DIR / "performance.csv")
    diagnostics = {}
    if RUN_SIGNAL_DIAGNOSTICS:
        # 即使策略总开关关闭 KAMA，也可独立检验 KAMA 过滤条件本身。
        diagnostic_kama_cache = kama_cache if kama_cache is not None else precompute_kama(close)
        diagnostics = export_signal_diagnostics(
            close, adj_open, diagnostic_kama_cache, momentum_cache)
    # 原程序允许关闭归因。此时保持上面的原回测调用不变，
    # 仅为交易指标补跑一遍逐资产记录，并核对净值完全一致。
    metric_audits = audits
    if not RUN_ATTRIBUTION:
        metric_audits = {}
        original_export_details = EXPORT_DETAILS
        try:
            # 补跑只在内存中取逐资产记录，不覆盖原有的明细文件。
            globals()["EXPORT_DETAILS"] = False
            for name, method, use_kama in strategy_specs:
                metric_audits[name] = {}
                metric_signal = make_momentum_signal(
                    method, use_kama=use_kama, kama_cache=kama_cache,
                    momentum_cache=momentum_cache)
                metric_ledger = backtest(
                    close, adj_open, name, metric_signal,
                    rebalance="monthly", gross_cap=1.0,
                    audit=metric_audits[name])
                if not metric_ledger.equals(results[name]):
                    raise AssertionError(f"{name}: metrics replay changed original NAV")
        finally:
            globals()["EXPORT_DETAILS"] = original_export_details
    # 单独生成课程所需的关键指标表，不改动原绩效表的字段。
    key_metrics = export_key_metrics(
        results, metric_audits, performance, diagnostics)
    print("\nKey metrics saved to key_metrics.csv and key_metrics.md:")
    print(key_metrics.to_string(float_format=lambda v: f"{v:.4f}"))
    names = [n for n in MOMENTUM_STRATEGY_ORDER + [
        "SPY", "QQQ", "EqualWeight", "LeveragedEqualWeight", "BIL"]
        if n in performance.index]
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
        for name in MOMENTUM_STRATEGY_ORDER:
            reports[name] = export_attribution(name, results[name], audits[name], signals[name])
        compare_attribution(reports)
    costs = export_cost_analysis(results, zero_cost_results)
    write_report(performance, yearly, reports, costs, results, diagnostics)
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

    # 单独展示四条动量策略，便于拆分 KAMA 与 EPO 的效果。
    comparison = [t for t in MOMENTUM_STRATEGY_ORDER + ["SPY", "QQQ", "EqualWeight", "BIL"]
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
    # 最后弹出完整版指标仪表板；同时保存 PNG，关闭弹窗也不会丢失图片。
    metrics_fig = plot_key_metrics(
        key_metrics, diagnostics, results["Momentum_kama"])
    print(f"完整版指标图已保存：{OUT_DIR / 'key_metrics.png'}")
    if SHOW_PLOTS:
        print("正在显示完整版指标图；关闭此窗口后程序结束。")
        plt.show()
    plt.close(metrics_fig)
    print(f"\n全部输出完成：{OUT_DIR}\n先阅读 REPORT_CN.md；EXPORT_DETAILS={EXPORT_DETAILS}。")
    return {"results": results, "audits": audits, "reports": reports,
            "zero_cost_results": zero_cost_results,
            "key_metrics": key_metrics, "signal_diagnostics": diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF momentum, KAMA and EPO backtest")
    parser.add_argument("--epo-example", action="store_true", help="Print two-asset EPO example")
    args = parser.parse_args()
    if args.epo_example:
        print_epo_example()
    else:
        main()

