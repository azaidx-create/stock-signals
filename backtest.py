from datetime import date
import json

import numpy as np
import pandas as pd
import yfinance as yf

from signal_check import WATCHLIST


START_DOWNLOAD = "2018-01-01"
TEST_START = pd.Timestamp("2020-01-01")
END = (date.today() + pd.Timedelta(days=1)).isoformat()


def download_one(ticker):
    try:
        data = yf.download(
            ticker,
            start=START_DOWNLOAD,
            end=END,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception as exc:
        print(f"DOWNLOAD_ERROR {ticker}: {exc}")
        return pd.DataFrame()


spy = download_one("^GSPC")
spy["market_sma200"] = spy["Close"].rolling(200).mean()
market_up = (spy["Close"] > spy["market_sma200"]).shift(1)

trades = []
missing = []

for number, ticker in enumerate(WATCHLIST, 1):
    data = download_one(ticker)
    if len(data) < 220:
        missing.append(ticker)
        continue

    data["sma50"] = data["Close"].rolling(50).mean()
    data["sma200"] = data["Close"].rolling(200).mean()
    distance = (data["Close"] - data["sma50"]) / data["sma50"]
    signal = (
        (data["sma50"] > data["sma200"])
        & distance.between(0, 0.05)
    )
    signal = signal.shift(1).fillna(False)
    signal &= market_up.reindex(data.index).fillna(False).astype(bool)

    in_position = False
    entry_price = None
    entry_date = None

    for dt, row in data.loc[data.index >= TEST_START].iterrows():
        if not in_position:
            if bool(signal.loc[dt]):
                entry_price = float(row["Open"])
                if not np.isfinite(entry_price) or entry_price <= 0:
                    continue
                entry_date = dt
                stop = entry_price * 0.90
                target = entry_price * 1.20
                in_position = True
            continue

        # Conservative assumption if both levels trade within one daily bar.
        if float(row["Low"]) <= stop:
            exit_price = stop
            outcome = "stop"
        elif float(row["High"]) >= target:
            exit_price = target
            outcome = "target"
        else:
            continue

        trades.append({
            "ticker": ticker,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": dt.strftime("%Y-%m-%d"),
            "entry": entry_price,
            "exit": exit_price,
            "return_pct": (exit_price / entry_price - 1) * 100,
            "days": (dt - entry_date).days,
            "outcome": outcome,
            "closed": True,
        })
        in_position = False

    if in_position:
        dt = data.index[-1]
        exit_price = float(data["Close"].iloc[-1])
        trades.append({
            "ticker": ticker,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": dt.strftime("%Y-%m-%d"),
            "entry": entry_price,
            "exit": exit_price,
            "return_pct": (exit_price / entry_price - 1) * 100,
            "days": (dt - entry_date).days,
            "outcome": "open_marked_to_market",
            "closed": False,
        })

    if number % 20 == 0:
        print(f"PROGRESS {number}/{len(WATCHLIST)}")

df = pd.DataFrame(trades)
closed = df[df["closed"]].copy()
recent = df[pd.to_datetime(df["entry_date"]) >= pd.Timestamp("2025-01-01")]

def summarize(frame):
    closed_frame = frame[frame["closed"]]
    returns = frame["return_pct"]
    return {
        "trades_including_open": int(len(frame)),
        "closed_trades": int(len(closed_frame)),
        "target_wins": int((closed_frame["outcome"] == "target").sum()),
        "stop_losses": int((closed_frame["outcome"] == "stop").sum()),
        "closed_win_rate_pct": float(
            (closed_frame["outcome"] == "target").mean() * 100
        ) if len(closed_frame) else None,
        "mean_trade_return_pct": float(returns.mean()) if len(frame) else None,
        "median_trade_return_pct": float(returns.median()) if len(frame) else None,
        "mean_holding_days": float(frame["days"].mean()) if len(frame) else None,
        "open_positions": int((~frame["closed"]).sum()),
    }

result = {
    "test_start": str(TEST_START.date()),
    "data_through": str(spy.index[-1].date()),
    "watchlist_size": len(WATCHLIST),
    "missing_tickers": missing,
    "all": summarize(df),
    "since_2025": summarize(recent),
}

print("RESULT_JSON")
print(json.dumps(result, indent=2))

with open("data/backtest_results.json", "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)
    file.write("\n")
