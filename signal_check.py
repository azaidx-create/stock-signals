"""
Automated Signal Check - writes a results page automatically.
------------------------------------------------------------------
Runs on GitHub's servers on a daily schedule (no button presses, no
phone needed). Checks the same tested strategy (50/200-day golden
cross, confirmed by S&P 500 trend, fundamentals-filtered) and writes
the results into docs/index.html. That file gets published as a real
website via GitHub Pages, so you just open one bookmarked link
anytime to see the latest check - already done for you.
"""

from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf

WATCHLIST = ["AAPL", "MSFT", "JNJ", "PG", "JPM", "KO", "WMT", "V", "HD", "COST"]
STOP_LOSS_PCT = -10.0


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def passes_fundamental_filter(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    earnings_growth = info.get("earningsGrowth")
    debt_to_equity = info.get("debtToEquity")
    profit_margin = info.get("profitMargins")
    if earnings_growth is not None and earnings_growth < 0:
        return False
    if debt_to_equity is not None and debt_to_equity > 200:
        return False
    if profit_margin is not None and profit_margin < 0:
        return False
    return True


def get_market_uptrend():
    spy = yf.download("^GSPC", period="2y", interval="1d", progress=False, auto_adjust=True)
    spy = _flatten_columns(spy)
    sma200 = spy["Close"].rolling(200).mean()
    return bool((spy["Close"].iloc[-1] > sma200.iloc[-1]).item())


def check_ticker(ticker):
    data = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    data = _flatten_columns(data)
    if data.empty or len(data) < 200:
        return None

    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    price = float(data["Close"].iloc[-1])
    sma50_today = float(data["SMA50"].iloc[-1])
    sma200_today = float(data["SMA200"].iloc[-1])
    sma50_yday = float(data["SMA50"].iloc[-2])
    sma200_yday = float(data["SMA200"].iloc[-2])

    crossed_today = (sma50_today > sma200_today) and (sma50_yday <= sma200_yday)
    in_uptrend = sma50_today > sma200_today

    return {
        "price": price, "sma50": sma50_today, "sma200": sma200_today,
        "crossed_today": crossed_today, "in_uptrend": in_uptrend,
    }


def build_html(market_up, rows, checked_at):
    buy_count = sum(1 for r in rows if r["status"] == "buy")

    if buy_count > 0:
        banner_bg, banner_text = "rgba(53,224,138,0.14)", f"{buy_count} buy signal(s) found today."
    else:
        banner_bg, banner_text = "rgba(138,148,141,0.1)", "No buy signals today. Nothing to act on."

    row_html = ""
    for r in rows:
        if r["status"] == "buy":
            badge = '<span style="background:#35e08a;color:#05130b;padding:6px 12px;border-radius:6px;font-weight:700;">BUY SIGNAL</span>'
        elif r["status"] == "wait_uptrend":
            badge = '<span style="color:#8a948d;">WAIT — already in uptrend</span>'
        elif r["status"] == "avoid":
            badge = '<span style="color:#e05555;">AVOID — below trend</span>'
        elif r["status"] == "skipped_fundamentals":
            badge = '<span style="color:#8a948d;">skipped — weak fundamentals</span>'
        else:
            badge = '<span style="color:#8a948d;">no data</span>'

        price_txt = f"${r['price']:.2f}" if r.get("price") is not None else "—"
        stop_txt = f"${r['price']*0.9:.2f}" if r["status"] == "buy" else "—"

        row_html += f"""
        <tr>
          <td style="padding:12px 10px;font-weight:700;border-bottom:1px solid #232a26;">{r['ticker']}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;">{price_txt}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;">{badge}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;color:#e0a835;">{stop_txt}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Desk — Daily Results</title>
<style>
  body{{ margin:0; background:#0a0d0c; color:#e7ebe6; font-family: -apple-system, sans-serif; padding: 24px 16px 50px; }}
  h1{{ font-size: 24px; margin-bottom: 4px; }}
  .sub{{ color:#8a948d; font-size:13px; margin-bottom:20px; }}
  .banner{{ padding:14px 16px; border-radius:6px; font-weight:700; margin-bottom:20px; background:{banner_bg}; }}
  table{{ width:100%; border-collapse: collapse; font-size:14px; }}
  th{{ text-align:left; font-size:11px; text-transform:uppercase; color:#8a948d; padding:10px; border-bottom:1px solid #232a26; }}
  footer{{ margin-top:30px; color:#8a948d; font-size:11.5px; line-height:1.6; }}
</style>
</head>
<body>
  <h1>Signal Desk — Daily Results</h1>
  <div class="sub">Last checked automatically: {checked_at}. Updates once per weekday — no need to press anything.</div>
  <div class="banner">{banner_text}</div>
  <div class="sub">S&amp;P 500 trend: <b>{'UPTREND' if market_up else 'DOWNTREND'}</b></div>
  <table>
    <thead><tr><th>Ticker</th><th>Price</th><th>Status</th><th>Stop-loss (-10%)</th></tr></thead>
    <tbody>{row_html}</tbody>
  </table>
  <footer>
    This reflects a strategy tested to show roughly a 55-60% win rate in normal markets,
    and losses during prolonged downturns like 2000-2002. Not financial advice.
    This page updates automatically once a day via a scheduled check - no action needed from you.
  </footer>
</body>
</html>"""


def main():
    market_up = get_market_uptrend()
    rows = []

    for ticker in WATCHLIST:
        fund_ok = passes_fundamental_filter(ticker)
        if fund_ok is False:
            rows.append({"ticker": ticker, "price": None, "status": "skipped_fundamentals"})
            continue

        result = check_ticker(ticker)
        if result is None:
            rows.append({"ticker": ticker, "price": None, "status": "no_data"})
            continue

        buy_signal = result["crossed_today"] and market_up
        if buy_signal:
            status = "buy"
        elif result["in_uptrend"]:
            status = "wait_uptrend"
        else:
            status = "avoid"

        rows.append({"ticker": ticker, "price": result["price"], "status": status})

    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(market_up, rows, checked_at)

    with open("docs/index.html", "w") as f:
        f.write(html)

    print("Wrote docs/index.html")


if __name__ == "__main__":
    main()
