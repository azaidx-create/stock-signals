from datetime import date
import json
import time
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

from signal_check import WATCHLIST


START_DOWNLOAD = "2018-01-01"
TEST_START = pd.Timestamp(date.today()) - pd.DateOffset(months=6)
END = (date.today() + pd.Timedelta(days=1)).isoformat()
SEC_USER_AGENT = (
    "stock-signals-backtest "
    "azaidx-create@users.noreply.github.com"
)
SEC_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}


def get_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": SEC_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def load_sec_ticker_map():
    raw = get_json("https://www.sec.gov/files/company_tickers.json")
    return {
        item["ticker"].upper().replace(".", "-"): int(item["cik_str"])
        for item in raw.values()
    }


def load_company_facts(cik):
    time.sleep(0.12)
    try:
        return get_json(
            "https://data.sec.gov/api/xbrl/companyfacts/"
            f"CIK{cik:010d}.json"
        )
    except Exception as exc:
        print(f"SEC_ERROR CIK {cik}: {exc}")
        return None


def concept_values(company_facts, tags, unit="USD"):
    facts = company_facts.get("facts", {}).get("us-gaap", {})
    values = []
    for tag in tags:
        units = facts.get(tag, {}).get("units", {})
        values.extend(units.get(unit, []))
    return values


def available_values(values, as_of):
    selected = []
    for item in values:
        if item.get("form") not in SEC_FORMS:
            continue
        try:
            filed = pd.Timestamp(item["filed"])
            end = pd.Timestamp(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if filed <= as_of and end <= as_of and np.isfinite(item.get("val", np.nan)):
            selected.append({**item, "_filed": filed, "_end": end})
    return selected


def latest_instant(company_facts, tags, as_of):
    values = available_values(concept_values(company_facts, tags), as_of)
    if not values:
        return None
    item = max(values, key=lambda x: (x["_end"], x["_filed"]))
    return float(item["val"]), item["_end"]


def quarterly_series(company_facts, tags, as_of):
    values = available_values(concept_values(company_facts, tags), as_of)
    by_end = {}
    for item in values:
        if "start" not in item:
            continue
        start = pd.Timestamp(item["start"])
        days = (item["_end"] - start).days
        if not 70 <= days <= 110:
            continue
        previous = by_end.get(item["_end"])
        if previous is None or item["_filed"] > previous["_filed"]:
            by_end[item["_end"]] = item
    return sorted(by_end.values(), key=lambda x: x["_end"])


def historical_fundamentals_pass(company_facts, as_of):
    """Approximate the live Yahoo filter with point-in-time SEC facts."""
    if not company_facts:
        return True, "unavailable"

    revenue = quarterly_series(
        company_facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ],
        as_of,
    )
    income = quarterly_series(
        company_facts,
        ["NetIncomeLoss", "ProfitLoss"],
        as_of,
    )

    revenue_by_end = {item["_end"]: float(item["val"]) for item in revenue}
    income_by_end = {item["_end"]: float(item["val"]) for item in income}
    common_ends = sorted(set(revenue_by_end) & set(income_by_end))

    if common_ends:
        latest_end = common_ends[-1]
        latest_revenue = revenue_by_end[latest_end]
        latest_income = income_by_end[latest_end]
        if latest_revenue > 0 and latest_income / latest_revenue < 0:
            return False, "negative_profit_margin"

        prior_candidates = [
            end for end in common_ends
            if 330 <= (latest_end - end).days <= 400
        ]
        if prior_candidates:
            prior_income = income_by_end[prior_candidates[-1]]
            if latest_income < prior_income:
                return False, "negative_earnings_growth"

    equity = latest_instant(
        company_facts,
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        as_of,
    )
    debt_current = latest_instant(
        company_facts,
        [
            "LongTermDebtCurrent",
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "ShortTermBorrowings",
        ],
        as_of,
    )
    debt_noncurrent = latest_instant(
        company_facts,
        [
            "LongTermDebtNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        ],
        as_of,
    )

    if equity and equity[0] > 0 and (debt_current or debt_noncurrent):
        total_debt = (debt_current[0] if debt_current else 0) + (
            debt_noncurrent[0] if debt_noncurrent else 0
        )
        if total_debt / equity[0] * 100 > 200:
            return False, "debt_to_equity_above_200"

    return True, "passed"


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
with open("data/sec_fundamental_history.json", encoding="utf-8") as file:
    sec_history = json.load(file)

SAMPLE = sorted(sec_history["tickers"])


def cached_fundamental_status(ticker, as_of):
    selected = None
    for item in sec_history["tickers"].get(ticker, []):
        if pd.Timestamp(item["date"]) <= as_of:
            selected = item
        else:
            break
    if selected is None:
        return True, "unavailable"
    return bool(selected["pass"]), selected["reason"]

trades = []
missing = []
fundamental_stats = {
    "candidates": 0,
    "passed": 0,
    "unavailable": 0,
    "negative_profit_margin": 0,
    "negative_earnings_growth": 0,
    "debt_to_equity_above_200": 0,
}

for number, ticker in enumerate(SAMPLE, 1):
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
    entry_event = signal & ~signal.shift(1).fillna(False)

    in_position = False
    entry_price = None
    entry_date = None
    for dt, row in data.loc[data.index >= TEST_START].iterrows():
        if not in_position:
            if bool(entry_event.loc[dt]):
                fundamental_stats["candidates"] += 1
                fund_ok, fund_reason = cached_fundamental_status(
                    ticker,
                    dt,
                )
                fundamental_stats[fund_reason] += 1
                if not fund_ok:
                    continue

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
        print(f"PROGRESS {number}/{len(SAMPLE)}")

df = pd.DataFrame(trades)
closed = df[df["closed"]].copy()
recent = df[pd.to_datetime(df["entry_date"]) >= TEST_START]

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
    "sample_size": len(SAMPLE),
    "missing_tickers": missing,
    "fundamental_method": (
        "Point-in-time SEC filings: latest reported quarterly net margin, "
        "quarterly net-income growth versus the comparable prior-year "
        "quarter, and reported debt/equity. Missing fields pass, matching "
        "the live scanner's behavior."
    ),
    "fundamental_stats": fundamental_stats,
    "all": summarize(df),
    "six_months": summarize(recent),
}

print("RESULT_JSON")
print(json.dumps(result, indent=2))

with open("data/backtest_results.json", "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)
    file.write("\n")
