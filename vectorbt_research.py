"""Research the signal strategy with SimFin fundamentals and VectorBT."""

from datetime import date, timedelta
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import simfin as sf
import vectorbt as vbt
import yfinance as yf
from simfin.names import (
    LT_DEBT,
    NET_INCOME,
    PUBLISH_DATE,
    REPORT_DATE,
    REVENUE,
    ST_DEBT,
    TICKER,
    TOTAL_EQUITY,
)

from signal_check import WATCHLIST


RESULT_FILE = Path("data/vectorbt_results.json")
SIMFIN_DIR = Path(".simfin_data")
YEARS = 5
FEES = 0.001
SLIPPAGE = 0.001


def load_simfin_dataset(loaders):
    frames = []
    for loader in loaders:
        try:
            frame = loader(
                variant="quarterly",
                market="us",
                refresh_days=30,
            ).reset_index()
            frames.append(frame)
        except Exception as error:
            print(f"SimFin dataset skipped: {error}")
    if not frames:
        raise RuntimeError("SimFin returned no fundamental datasets.")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_fundamentals():
    api_key = os.environ.get("SIMFIN_API_KEY")
    if not api_key:
        raise RuntimeError("SIMFIN_API_KEY is missing.")

    sf.set_api_key(api_key)
    sf.set_data_dir(str(SIMFIN_DIR))

    income = load_simfin_dataset(
        [sf.load_income, sf.load_income_banks, sf.load_income_insurance]
    )
    balance = load_simfin_dataset(
        [sf.load_balance, sf.load_balance_banks, sf.load_balance_insurance]
    )

    income_cols = [
        column for column in
        [TICKER, REPORT_DATE, PUBLISH_DATE, REVENUE, NET_INCOME]
        if column in income.columns
    ]
    balance_cols = [
        column for column in
        [TICKER, REPORT_DATE, PUBLISH_DATE, ST_DEBT, LT_DEBT, TOTAL_EQUITY]
        if column in balance.columns
    ]

    merged = income[income_cols].merge(
        balance[balance_cols],
        on=[TICKER, REPORT_DATE],
        how="outer",
        suffixes=("_income", "_balance"),
    )

    publish_columns = [
        column for column in merged.columns
        if column.startswith(PUBLISH_DATE)
    ]
    merged["Known Date"] = merged[publish_columns].max(axis=1)
    merged.sort_values([TICKER, REPORT_DATE], inplace=True)

    def numeric_column(name):
        if name not in merged:
            return pd.Series(np.nan, index=merged.index)
        return pd.to_numeric(merged[name], errors="coerce")

    net_income = numeric_column(NET_INCOME)
    revenue = numeric_column(REVENUE)
    equity = numeric_column(TOTAL_EQUITY)
    merged["Profit Margin"] = net_income / revenue
    prior_income = net_income.groupby(merged[TICKER]).shift(4)
    merged["Earnings Growth"] = (
        net_income - prior_income
    ) / prior_income.abs()
    total_debt = numeric_column(ST_DEBT).fillna(0) + numeric_column(
        LT_DEBT
    ).fillna(0)
    merged["Debt To Equity"] = total_debt / equity * 100

    merged["Fundamental Pass"] = ~(
        (merged["Earnings Growth"].notna() & (merged["Earnings Growth"] < 0))
        | (merged["Profit Margin"].notna() & (merged["Profit Margin"] < 0))
        | (merged["Debt To Equity"].notna() & (merged["Debt To Equity"] > 200))
    )

    return merged.dropna(subset=[TICKER, "Known Date"])


def download_prices(ticker, start):
    try:
        data = yf.download(
            ticker,
            start=start,
            end=date.today() + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception as error:
        print(f"Price download failed for {ticker}: {error}")
        return pd.DataFrame()


def fundamental_mask(ticker, index, fundamentals):
    rows = fundamentals[fundamentals[TICKER] == ticker][
        ["Known Date", "Fundamental Pass"]
    ].copy()
    rows["Known Date"] = pd.to_datetime(
        rows["Known Date"]
    ).astype("datetime64[ns]")
    rows.sort_values("Known Date", inplace=True)
    if rows.empty:
        return pd.Series(True, index=index)

    daily = pd.DataFrame({
        "Date": pd.to_datetime(pd.DatetimeIndex(index)).astype("datetime64[ns]")
    })
    known = pd.merge_asof(
        daily.sort_values("Date"),
        rows.rename(columns={"Known Date": "Date"}),
        on="Date",
        direction="backward",
    )
    return pd.Series(
        known["Fundamental Pass"].fillna(True).to_numpy(dtype=bool),
        index=index,
    )


def summarize_returns(returns):
    values = np.asarray(returns, dtype=float)
    if not len(values):
        return {"closed_trades": 0, "win_rate_pct": None, "mean_return_pct": None}
    return {
        "closed_trades": int(len(values)),
        "win_rate_pct": float((values > 0).mean() * 100),
        "mean_return_pct": float(values.mean() * 100),
        "median_return_pct": float(np.median(values) * 100),
        "profit_factor": float(
            values[values > 0].sum() / abs(values[values < 0].sum())
        ) if (values < 0).any() else None,
    }


def main():
    fundamentals = load_fundamentals()
    covered = sorted(set(WATCHLIST) & set(fundamentals[TICKER]))
    start = date.today() - timedelta(days=365 * YEARS + 300)

    market = download_prices("^GSPC", start)
    market_up = market["Close"] > market["Close"].rolling(200).mean()

    price_data = {}
    for number, ticker in enumerate(covered, 1):
        data = download_prices(ticker, start)
        if len(data) >= 220:
            price_data[ticker] = data
        if number % 25 == 0:
            print(f"Price progress: {number}/{len(covered)}")

    configurations = []
    for max_distance in [0.03, 0.05, 0.08]:
        for stop in [0.08, 0.10, 0.12]:
            for target in [0.15, 0.20, 0.25]:
                returns = []
                for ticker, data in price_data.items():
                    close = pd.Series(
                        data["Close"].to_numpy(dtype=np.float64),
                        index=data.index,
                    )
                    open_price = pd.Series(
                        data["Open"].to_numpy(dtype=np.float64),
                        index=data.index,
                    )
                    high = pd.Series(
                        data["High"].to_numpy(dtype=np.float64),
                        index=data.index,
                    )
                    low = pd.Series(
                        data["Low"].to_numpy(dtype=np.float64),
                        index=data.index,
                    )
                    sma50 = close.rolling(50).mean()
                    sma200 = close.rolling(200).mean()
                    distance = (close - sma50) / sma50
                    fund_ok = fundamental_mask(ticker, data.index, fundamentals)
                    market_ok = market_up.reindex(data.index).ffill().fillna(False)
                    setup = (
                        fund_ok
                        & market_ok
                        & (sma50 > sma200)
                        & distance.between(0, max_distance)
                    )
                    entries = (
                        setup.astype(bool)
                        & ~setup.shift(1).fillna(False).astype(bool)
                    ).shift(1).fillna(False).astype(bool)

                    portfolio = vbt.Portfolio.from_signals(
                        close=close,
                        entries=entries,
                        exits=None,
                        open=open_price,
                        high=high,
                        low=low,
                        sl_stop=stop,
                        tp_stop=target,
                        fees=FEES,
                        slippage=SLIPPAGE,
                        freq="1D",
                    )
                    returns.extend(portfolio.trades.closed.returns.values.tolist())

                configurations.append({
                    "max_distance_pct": max_distance * 100,
                    "stop_pct": stop * 100,
                    "target_pct": target * 100,
                    **summarize_returns(returns),
                })

    configurations.sort(
        key=lambda row: (
            row.get("profit_factor") or -1,
            row.get("mean_return_pct") or -999,
        ),
        reverse=True,
    )
    baseline = next(
        row for row in configurations
        if row["max_distance_pct"] == 5
        and row["stop_pct"] == 10
        and row["target_pct"] == 20
    )

    result = {
        "generated": str(date.today()),
        "years": YEARS,
        "watchlist_size": len(WATCHLIST),
        "simfin_covered": len(covered),
        "price_covered": len(price_data),
        "fees_pct_each_side": FEES * 100,
        "slippage_pct_each_side": SLIPPAGE * 100,
        "baseline": baseline,
        "top_10_configurations": configurations[:10],
        "all_configurations": configurations,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
