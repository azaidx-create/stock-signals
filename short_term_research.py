"""Walk-forward research for short-duration, high-hit-rate stock signals.

This deliberately tests a small, locked set of strategy hypotheses. Signals are
formed after a daily close, filled at the next open, and held for no more than
three sessions. The newest two years are never used to rank the candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from signal_check import WATCHLIST


RESULT_FILE = Path("data/short_term_research.json")
YEARS = 8
TEST_YEARS = 2
ROUND_TRIP_COST_PCT = 0.40
MAX_SIGNALS_PER_WEEK = 2


@dataclass(frozen=True)
class Strategy:
    name: str
    setup: str
    target_pct: float
    stop_pct: float
    max_holding_days: int


# Keep this list small. Adding configurations after seeing the test results
# invalidates the untouched test and requires moving the test window forward.
STRATEGIES = (
    Strategy("rsi2_balanced", "rsi2", 2.5, 3.0, 3),
    Strategy("rsi2_fast", "rsi2", 1.8, 2.8, 2),
    Strategy("three_day_pullback", "three_day", 2.5, 3.0, 3),
    Strategy("strict_rsi2", "strict_rsi2", 1.5, 2.0, 2),
    Strategy("ranked_pullback", "ranked_pullback", 2.0, 2.5, 3),
    Strategy("ranked_momentum", "ranked_momentum", 3.0, 2.5, 5),
)


def download(ticker: str, start: date) -> pd.DataFrame:
    try:
        frame = yf.download(
            ticker,
            start=start,
            end=date.today() + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as error:
        print(f"Download failed for {ticker}: {error}")
        return pd.DataFrame()


def rsi(close: pd.Series, periods: int) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / periods, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / periods, adjust=False).mean()
    strength = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + strength)


def enrich(frame: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"]
    data["sma50"] = close.rolling(50).mean()
    data["sma200"] = close.rolling(200).mean()
    data["rsi2"] = rsi(close, 2)
    data["return3"] = close.pct_change(3)
    data["return21"] = close.pct_change(21)
    data["return63"] = close.pct_change(63)
    data["avg_dollar_volume"] = (close * data["Volume"]).rolling(20).mean()
    spy_close = spy["Close"].reindex(data.index).ffill()
    data["spy_return63"] = spy_close.pct_change(63)
    data["market_up"] = (
        spy_close > spy_close.rolling(200).mean()
    ).fillna(False)
    return data


def candidates(ticker: str, data: pd.DataFrame, setup: str) -> pd.DataFrame:
    common = (
        data["market_up"]
        & (data["Close"] > data["sma200"])
        & (data["sma50"] > data["sma200"])
        & (data["return63"] > data["spy_return63"])
        & (data["avg_dollar_volume"] >= 50_000_000)
    )
    if setup == "rsi2":
        selected = common & (data["rsi2"] <= 10) & (data["return3"] < 0)
        score = -data["rsi2"] + 100 * (data["return63"] - data["spy_return63"])
    elif setup == "three_day":
        selected = common & (data["return3"] <= -0.03) & (data["rsi2"] <= 25)
        score = -100 * data["return3"] + 100 * (
            data["return63"] - data["spy_return63"]
        )
    elif setup == "strict_rsi2":
        selected = common & (data["rsi2"] <= 5) & (data["return3"] <= -0.02)
        score = -data["rsi2"] + 100 * (
            data["return63"] - data["spy_return63"]
        )
    elif setup == "ranked_pullback":
        selected = common & (data["rsi2"] <= 15) & (data["return3"] <= -0.01)
        score = -100 * data["return3"] + 100 * (
            data["return63"] - data["spy_return63"]
        )
    elif setup == "ranked_momentum":
        selected = common & (data["return21"] > 0) & (data["return3"] > -0.015)
        score = 100 * (data["return63"] - data["spy_return63"]) + 50 * data["return21"]
    else:
        raise ValueError(f"Unknown setup: {setup}")

    result = pd.DataFrame({
        "signal_date": data.index,
        "ticker": ticker,
        "score": score,
    })
    return result.loc[selected.fillna(False).to_numpy()].dropna()


def select_weekly(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    selected = raw.copy()
    selected["week"] = selected["signal_date"].dt.to_period("W-FRI")
    return (
        selected.sort_values(["week", "score"], ascending=[True, False])
        .groupby("week", group_keys=False)
        .head(MAX_SIGNALS_PER_WEEK)
        .drop(columns="week")
        .sort_values("signal_date")
    )


def simulate(signal: pd.Series, data: pd.DataFrame, strategy: Strategy):
    locations = data.index.get_indexer([signal["signal_date"]])
    if not len(locations) or locations[0] < 0:
        return None
    entry_location = locations[0] + 1
    if entry_location >= len(data):
        return None

    entry_date = data.index[entry_location]
    entry = float(data["Open"].iloc[entry_location])
    signal_close = float(data["Close"].iloc[locations[0]])
    # Cancel a stale alert if the next open gaps too far in either direction.
    if abs(entry / signal_close - 1) > 0.015:
        return None

    target = entry * (1 + strategy.target_pct / 100)
    stop = entry * (1 - strategy.stop_pct / 100)
    last_location = min(
        entry_location + strategy.max_holding_days - 1,
        len(data) - 1,
    )
    exit_price = None
    exit_reason = "time"
    exit_date = data.index[last_location]

    for location in range(entry_location, last_location + 1):
        low = float(data["Low"].iloc[location])
        high = float(data["High"].iloc[location])
        # Daily bars do not reveal which was touched first; assuming the stop
        # wins ties is deliberately conservative.
        if low <= stop:
            exit_price = stop
            exit_reason = "stop"
            exit_date = data.index[location]
            break
        if high >= target:
            exit_price = target
            exit_reason = "target"
            exit_date = data.index[location]
            break

    if exit_price is None:
        exit_price = float(data["Close"].iloc[last_location])

    gross_return_pct = (exit_price / entry - 1) * 100
    return {
        "ticker": signal["ticker"],
        "signal_date": signal["signal_date"],
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry,
        "target_price": target,
        "stop_price": stop,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "return_pct": gross_return_pct - ROUND_TRIP_COST_PCT,
    }


def wilson_interval(wins: int, count: int) -> list[float | None]:
    if count == 0:
        return [None, None]
    z = 1.96
    proportion = wins / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    margin = z * sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    ) / denominator
    return [(centre - margin) * 100, (centre + margin) * 100]


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    returns = trades["return_pct"].astype(float)
    wins = returns > 0
    gross_profit = returns[wins].sum()
    gross_loss = -returns[~wins].sum()
    ordered = trades.sort_values("exit_date")["return_pct"] / 100
    # Equal notional per signal; this is a trade-sequence stress measure, not
    # a claim about a fully capital-constrained portfolio.
    equity = (1 + ordered).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "trades": int(len(trades)),
        "wins": int(wins.sum()),
        "win_rate_pct": float(wins.mean() * 100),
        "win_rate_95pct_interval": wilson_interval(int(wins.sum()), len(trades)),
        "mean_return_pct": float(returns.mean()),
        "median_return_pct": float(returns.median()),
        "average_winner_pct": float(returns[wins].mean()) if wins.any() else None,
        "average_loser_pct": float(returns[~wins].mean()) if (~wins).any() else None,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "trade_sequence_max_drawdown_pct": float(drawdown.min() * 100),
        "target_exits": int((trades["exit_reason"] == "target").sum()),
        "stop_exits": int((trades["exit_reason"] == "stop").sum()),
        "time_exits": int((trades["exit_reason"] == "time").sum()),
    }


def main() -> None:
    start = date.today() - timedelta(days=365 * YEARS + 300)
    cutoff = pd.Timestamp(date.today() - timedelta(days=365 * TEST_YEARS))
    spy = download("SPY", start)
    if len(spy) < 500:
        raise RuntimeError("Insufficient SPY history.")

    prices = {}
    for number, ticker in enumerate(WATCHLIST, 1):
        frame = download(ticker, start)
        if len(frame) >= 250:
            prices[ticker] = enrich(frame, spy)
        if number % 25 == 0:
            print(f"Price progress: {number}/{len(WATCHLIST)}")

    results = []
    for strategy in STRATEGIES:
        raw_frames = [
            candidates(ticker, frame, strategy.setup)
            for ticker, frame in prices.items()
        ]
        chosen = select_weekly(pd.concat(raw_frames, ignore_index=True))
        simulated = []
        for _, signal in chosen.iterrows():
            trade = simulate(signal, prices[signal["ticker"]], strategy)
            if trade:
                simulated.append(trade)
        trades = pd.DataFrame(simulated)
        training = trades[trades["signal_date"] < cutoff] if not trades.empty else trades
        testing = trades[trades["signal_date"] >= cutoff] if not trades.empty else trades
        results.append({
            "strategy": asdict(strategy),
            "training": summarize(training),
            "out_of_sample": summarize(testing),
            "passes_research_gate": bool(
                len(testing) >= 50
                and (testing["return_pct"] > 0).mean() >= 0.60
                and (summarize(testing).get("profit_factor") or 0) >= 1.30
            ),
        })

    payload = {
        "generated": str(date.today()),
        "method": "locked short-duration strategy tournament",
        "history_years": YEARS,
        "out_of_sample_start": str(cutoff.date()),
        "watchlist_size": len(WATCHLIST),
        "price_covered": len(prices),
        "max_signals_per_week": MAX_SIGNALS_PER_WEEK,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "gate": {
            "minimum_out_of_sample_trades": 50,
            "minimum_win_rate_pct": 60,
            "minimum_profit_factor": 1.30,
            "note": "Passing is necessary for paper trading, not proof of future profit.",
        },
        "limitations": [
            "Current watchlist creates survivorship bias.",
            "Daily bars cannot reveal intraday stop/target order; ties count as stops.",
            "Trade-sequence drawdown is not a capital-constrained portfolio drawdown.",
            "Yahoo data is suitable for research, not live execution.",
        ],
        "results": results,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
