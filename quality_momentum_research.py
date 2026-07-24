"""Point-in-time quality/value plus momentum portfolio research."""

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from signal_check import WATCHLIST
from vectorbt_research import load_fundamentals


RESULT_FILE = Path("data/quality_momentum_research.json")
START = date.today() - timedelta(days=365 * 9)
FEE_ROUND_TRIP = 0.002
TOP_N = 10


def prices(ticker):
    try:
        frame = yf.download(ticker, start=START, end=date.today() + timedelta(days=1), auto_adjust=True, progress=False, threads=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame["Close"].dropna()
    except Exception as error:
        print(f"Price download failed for {ticker}: {error}")
        return pd.Series(dtype=float)


def fundamentals_at(fundamentals, ticker, when):
    rows = fundamentals[fundamentals["Ticker"] == ticker]
    rows = rows[pd.to_datetime(rows["Known Date"]) <= when]
    if rows.empty:
        return None
    return rows.sort_values("Known Date").iloc[-1]


def run(close, fundamentals, start, end):
    monthly = close.resample("ME").last()
    spy = monthly["SPY"]
    months = monthly.index[(monthly.index >= pd.Timestamp(start)) & (monthly.index < pd.Timestamp(end))]
    returns = []
    spy_returns = []
    holdings = []
    for i in range(len(months) - 1):
        signal_date, next_date = months[i], months[i + 1]
        spy_sma = close["SPY"].rolling(200).mean().loc[:signal_date].iloc[-1]
        market_price = close["SPY"].asof(signal_date)
        market_ok = float(market_price) > float(spy_sma)
        if market_ok:
            rows = []
            for ticker in WATCHLIST:
                if ticker not in close:
                    continue
                row = fundamentals_at(fundamentals, ticker, signal_date)
                if row is None:
                    continue
                series = close[ticker].loc[:signal_date]
                if len(series) < 260:
                    continue
                momentum_12m = float(series.iloc[-1] / series.iloc[-253] - 1)
                momentum_3m = float(series.iloc[-1] / series.iloc[-64] - 1)
                margin = float(row.get("Profit Margin", np.nan))
                growth = float(row.get("Earnings Growth", np.nan))
                debt = float(row.get("Debt To Equity", np.nan))
                if not np.isfinite([momentum_12m, momentum_3m]).all():
                    continue
                # Missing fundamentals do not receive a quality score.
                if not np.isfinite([margin, growth, debt]).all():
                    continue
                if margin <= 0 or growth <= 0 or debt > 200:
                    continue
                rows.append((ticker, momentum_12m, momentum_3m, margin, growth, debt))
            table = pd.DataFrame(rows, columns=["ticker", "mom12", "mom3", "margin", "growth", "debt"])
            if not table.empty:
                for column in ["mom12", "mom3", "margin", "growth"]:
                    table[f"rank_{column}"] = table[column].rank(pct=True)
                table["rank_debt"] = 1 - table["debt"].rank(pct=True)
                table["score"] = table[["rank_mom12", "rank_mom3", "rank_margin", "rank_growth", "rank_debt"]].mean(axis=1)
                selected = table.sort_values("score", ascending=False).head(TOP_N)["ticker"].tolist()
            else:
                selected = ["SHY"]
        else:
            selected = ["SHY"]
        selected = [ticker for ticker in selected if ticker in monthly.columns]
        if not selected:
            continue
        portfolio_return = float(
            (monthly.loc[next_date, selected] / monthly.loc[signal_date, selected] - 1).mean()
        )
        benchmark_return = float(spy.loc[next_date] / spy.loc[signal_date] - 1)
        returns.append(portfolio_return - FEE_ROUND_TRIP)
        spy_returns.append(benchmark_return - FEE_ROUND_TRIP)
        holdings.append(selected)
    if not returns:
        return {"months": 0}
    result = pd.Series(returns, index=months[1:1 + len(returns)])
    benchmark = pd.Series(spy_returns, index=result.index)
    curve, spy_curve = (1 + result).cumprod(), (1 + benchmark).cumprod()
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1 / 365.25)
    return {
        "months": len(result),
        "annualized_return_pct": float((curve.iloc[-1] ** (1 / years) - 1) * 100),
        "spy_annualized_return_pct": float((spy_curve.iloc[-1] ** (1 / years) - 1) * 100),
        "max_drawdown_pct": float((curve / curve.cummax() - 1).min() * 100),
        "spy_max_drawdown_pct": float((spy_curve / spy_curve.cummax() - 1).min() * 100),
        "sharpe_annualized": float(result.mean() / result.std() * np.sqrt(12)) if result.std() else None,
        "winning_months_pct": float((result > 0).mean() * 100),
        "final_holdings": holdings[-1],
    }


def main():
    fundamentals = load_fundamentals()
    tickers = sorted(set(WATCHLIST) | {"SPY", "SHY"})
    close = pd.concat({ticker: prices(ticker) for ticker in tickers}, axis=1).sort_index().ffill()
    close = close.dropna(axis=1, thresh=500)
    first = close.index.min().to_period("M").end_time.date()
    windows = []
    start = pd.Timestamp(first)
    while start + pd.DateOffset(years=2) <= pd.Timestamp(date.today()):
        end = start + pd.DateOffset(years=2)
        metrics = run(close, fundamentals, start.date(), end.date())
        metrics.update(period_start=str(start.date()), period_end=str(end.date()))
        windows.append(metrics)
        start = end
    result = {
        "generated": str(date.today()),
        "top_n": TOP_N,
        "price_covered": len(close.columns),
        "rolling_two_year_windows": windows,
        "note": "Point-in-time SimFin fundamentals; monthly equal-weight portfolio; no live signals generated.",
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
