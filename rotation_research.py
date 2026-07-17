"""FinRL-X-inspired weekly rotation benchmark.

This is a transparent reproduction of the strategy ideas (regime filter,
residual momentum, weekly rebalance, and defensive fallback), not a claim that
it is the original repository's exact implementation.
"""

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


RESULT_FILE = Path("data/rotation_research.json")
START = date.today() - timedelta(days=365 * 9)
FEE_EACH_SIDE = 0.001

GROUPS = {
    "growth": ["QQQ", "XLK", "XLY", "XLC", "IWM"],
    "real_assets": ["XLE", "XLB", "GLD", "VNQ", "DBC"],
    "defensive": ["XLU", "XLP", "XLV", "TLT", "SHY"],
}
ALL_TICKERS = sorted({ticker for values in GROUPS.values() for ticker in values} | {"SPY", "^VIX"})


def download(ticker):
    try:
        frame = yf.download(
            ticker,
            start=START,
            end=date.today() + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame[["Open", "Close"]].dropna()
    except Exception as error:
        print(f"Download failed for {ticker}: {error}")
        return pd.DataFrame()


def weekly_returns(prices):
    # Friday close-to-Friday close; the signal is shifted one week before use.
    return prices.resample("W-FRI").last().pct_change()


def run(prices, start_date, end_date):
    close = prices
    spy = close["SPY"]
    vix = close["^VIX"]
    weekly = weekly_returns(close)
    spy_weekly = weekly["SPY"]
    spy_sma = spy.rolling(200).mean().resample("W-FRI").last()
    vix_weekly = vix.resample("W-FRI").last()
    vix_threshold = vix_weekly.rolling(252, min_periods=60).quantile(0.80)
    regime_ok = (spy.resample("W-FRI").last() > spy_sma) & (vix_weekly <= vix_threshold)

    equity = 1.0
    benchmark = 1.0
    rows = []
    for i in range(64, len(weekly.index) - 1):
        signal_week = weekly.index[i]
        trade_week = weekly.index[i + 1]
        if signal_week < pd.Timestamp(start_date) or signal_week >= pd.Timestamp(end_date):
            continue
        scores = {}
        for group, tickers in GROUPS.items():
            group_scores = {}
            for ticker in tickers:
                if ticker not in close:
                    continue
                series = close[ticker]
                residual = series.pct_change(63) - spy.pct_change(63)
                residual_window = residual.loc[:signal_week].tail(63).dropna()
                if len(residual_window) < 30:
                    continue
                information_ratio = residual_window.mean() / residual_window.std()
                momentum = series.pct_change(63).loc[:signal_week].iloc[-1]
                group_scores[ticker] = information_ratio + momentum
            if group_scores:
                scores[group] = max(group_scores, key=group_scores.get)

        if not scores:
            continue
        risk_on = bool(regime_ok.get(signal_week, False))
        if not risk_on:
            selected = ["SHY"]
        else:
            # At most two active groups, chosen by their selected asset's score.
            ranked = []
            for group, ticker in scores.items():
                series = close[ticker]
                residual = series.pct_change(63) - spy.pct_change(63)
                window = residual.loc[:signal_week].tail(63).dropna()
                ranked.append((window.mean() / window.std() + series.pct_change(63).loc[:signal_week].iloc[-1], ticker))
            selected = [ticker for _, ticker in sorted(ranked, reverse=True)[:2]]

        next_returns = weekly.loc[trade_week, selected].dropna()
        if next_returns.empty:
            continue
        gross = float(next_returns.mean())
        cost = FEE_EACH_SIDE * 2 * (len(selected) / max(len(scores), 1))
        net = gross - cost
        spy_net = float(spy_weekly.get(trade_week, 0.0)) - FEE_EACH_SIDE * 2
        equity *= 1 + net
        benchmark *= 1 + spy_net
        rows.append({"date": str(trade_week.date()), "return": net, "spy_return": spy_net, "selected": selected, "risk_on": risk_on})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"weeks": 0}
    returns = frame["return"]
    spy_returns = frame["spy_return"]
    curve = (1 + returns).cumprod()
    spy_curve = (1 + spy_returns).cumprod()
    years = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25, 1 / 365.25)
    return {
        "weeks": int(len(frame)),
        "annualized_return_pct": float((equity ** (1 / years) - 1) * 100),
        "spy_annualized_return_pct": float((benchmark ** (1 / years) - 1) * 100),
        "total_return_pct": float((equity - 1) * 100),
        "spy_total_return_pct": float((benchmark - 1) * 100),
        "max_drawdown_pct": float((curve / curve.cummax() - 1).min() * 100),
        "spy_max_drawdown_pct": float((spy_curve / spy_curve.cummax() - 1).min() * 100),
        "sharpe_annualized": float(returns.mean() / returns.std() * np.sqrt(52)) if returns.std() else None,
        "spy_sharpe_annualized": float(spy_returns.mean() / spy_returns.std() * np.sqrt(52)) if spy_returns.std() else None,
        "winning_weeks_pct": float((returns > 0).mean() * 100),
        "spy_winning_weeks_pct": float((spy_returns > 0).mean() * 100),
        "final_holdings": frame.iloc[-1]["selected"],
    }


def main():
    frames = {ticker: download(ticker) for ticker in ALL_TICKERS}
    covered = {ticker: frame for ticker, frame in frames.items() if len(frame) > 500}
    prices = pd.concat({ticker: frame["Close"] for ticker, frame in covered.items()}, axis=1).sort_index().ffill()
    test_start = date.today() - timedelta(days=365 * 2)
    train_end = test_start
    first_week = prices.index.min().to_period("W-FRI").start_time.date()
    windows = []
    window_start = pd.Timestamp(first_week)
    while window_start + pd.DateOffset(years=2) <= pd.Timestamp(date.today()):
        window_end = window_start + pd.DateOffset(years=2)
        metrics = run(prices, window_start.date(), window_end.date())
        metrics["period_start"] = str(window_start.date())
        metrics["period_end"] = str(window_end.date())
        windows.append(metrics)
        window_start = window_end
    result = {
        "generated": str(date.today()),
        "covered": sorted(covered),
        "groups": GROUPS,
        "fees_each_side_pct": FEE_EACH_SIDE * 100,
        "training": run(prices, START, train_end),
        "out_of_sample": run(prices, test_start, date.today()),
        "rolling_two_year_windows": windows,
        "rolling_summary": {
            "windows_tested": len(windows),
            "windows_beating_spy": sum(
                row.get("annualized_return_pct", -999)
                > row.get("spy_annualized_return_pct", 999)
                for row in windows
            ),
            "windows_with_positive_return": sum(
                row.get("annualized_return_pct", -999) > 0
                for row in windows
            ),
        },
        "gate": "Out-of-sample strategy must beat SPY annualized return, have lower drawdown, and positive Sharpe.",
        "limitation": "This is FinRL-X-inspired, not an exact reproduction; ETF survivorship and data-source differences remain.",
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
