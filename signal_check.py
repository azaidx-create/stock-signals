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

from datetime import datetime, timezone, timedelta
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import numpy as np
import yfinance as yf

SAUDI_OFFSET = timedelta(hours=3)  # Saudi Arabia is UTC+3, no daylight saving

WATCHLIST = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","JNJ",
    "V","PG","XOM","HD","MA","CVX","ABBV","PFE","KO","PEP","MRK","WMT",
    "BAC","COST","DIS","CSCO","TMO","MCD","ABT","CRM","ACN","DHR","NKE",
    "LIN","TXN","NEE","PM","UPS","RTX","UNP","MS","SPGI","LOW","INTC",
    "HON","IBM","CAT","GS","AMGN","SBUX","BA","GE","DE","BLK","AXP","LMT",
    "MDT","PLD","SYK","GILD","MMC","ADI","CI","T","SO","MO","BKNG","TJX",
    "REGN","VRTX","ZTS","CB","PGR","ETN","BSX","CME","DUK","SLB","ITW",
    "APD","EOG","AON","CL","SHW","NSC","WM","EMR","ORLY","MCO","ADP",
    "PANW","FDX","GD","NOC","MET","USB","SCHW","TGT","PSX","COP","MDLZ",
    "KMB","AMT","BDX","BK","C","CMCSA","CSX","CVS","D","DOW","AEP","TRV",
    "WFC","F","GM","AMD","AVGO","ORCL","QCOM","ADBE","NFLX","PYPL",
    "INTU","NOW","UBER","LRCX","KLAC","SNPS","CDNS","MU","APH","ANET",
    "ROP","FTNT","MSI","CTAS","PAYX","VRSK","FAST","ODFL","EW","IDXX",
    "A","IQV","HUM","CNC","ELV","MRNA","BIIB","ALGN","DXCM","ISRG","HCA",
    "PNC","TFC","COF","AIG","AFL","ALL","PRU","TROW","STT","NTRS","MTB",
    "FITB","HBAN","RF","KEY","CFG","SYF","DFS","ALLY","NDAQ","ICE","MCK",
]
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
    crossed_down_today = (sma50_today < sma200_today) and (sma50_yday >= sma200_yday)
    in_uptrend = sma50_today > sma200_today

    latest_data_date = data.index[-1].strftime("%Y-%m-%d")

    return {
        "price": price, "sma50": sma50_today, "sma200": sma200_today,
        "crossed_today": crossed_today, "crossed_down_today": crossed_down_today,
        "in_uptrend": in_uptrend, "latest_data_date": latest_data_date,
    }



def send_telegram_message(message):
    """Send a Telegram alert using GitHub Actions repository secrets."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("Telegram skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    request = urllib.request.Request(url, data=payload, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

        if result.get("ok"):
            print("Telegram notification sent successfully.")
            return True

        print(f"Telegram rejected the message: {result}")
        return False
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"Telegram HTTP error {error.code}: {details}")
        return False
    except Exception as error:
        print(f"Telegram notification failed: {error}")
        return False


def build_telegram_message(buy_rows, checked_at):
    """Create one Telegram message containing all BUY signals."""
    lines = [
        "📈 STOCK BUY SIGNAL",
        "",
        f"Checked: {checked_at}",
        f"Signals found: {len(buy_rows)}",
        "",
    ]

    for row in buy_rows:
        price = row["price"]
        lines.extend([
            f"🟢 {row['ticker']}",
            f"Buy around: ${price:.2f}",
            f"Stop-loss: ${price * 0.90:.2f} (-10%)",
            f"Take profit: ${price * 1.20:.2f} (+20%)",
            "Reason: New 50/200-day golden cross",
            "Market filter: S&P 500 uptrend confirmed",
            "",
        ])

    lines.append("Review the price in TradingView before placing a paper trade.")
    return "\n".join(lines)


def build_html(market_up, rows, checked_at, run_type, stats, latest_data_date):
    buy_count = stats["buy"]

    if buy_count > 0:
        banner_bg = "rgba(53,224,138,0.14)"
        banner_text = f"{buy_count} buy signal(s) found today."
    else:
        banner_bg = "rgba(138,148,141,0.1)"
        banner_text = "No buy signals today. Nothing to act on."

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
        if r["status"] == "buy":
            entry_txt = f'<b style="color:#35e08a;">${r["price"]:.2f}</b>'
            stop_txt = f"${r['price']*0.9:.2f}"
            target_txt = f'<b style="color:#4ba3ff;">${r["price"]*1.20:.2f}</b>'
        else:
            entry_txt = "—"
            stop_txt = "—"
            target_txt = "—"

        row_html += f"""
        <tr>
          <td style="padding:12px 10px;font-weight:700;border-bottom:1px solid #232a26;">{r['ticker']}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;">{price_txt}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;">{badge}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;">{entry_txt}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;color:#e0a835;">{stop_txt}</td>
          <td style="padding:12px 10px;border-bottom:1px solid #232a26;">{target_txt}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Desk — Daily Results</title>
<style>
  body{{ margin:0; background:#0a0d0c; color:#e7ebe6; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; padding:24px 16px 50px; }}
  h1{{ font-size:24px; margin-bottom:4px; }}
  .sub{{ color:#8a948d; font-size:13px; margin-bottom:16px; }}
  .banner{{ padding:14px 16px; border-radius:6px; font-weight:700; margin-bottom:18px; background:{banner_bg}; }}
  .stats{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; margin:0 0 22px; }}
  .card{{ background:#111613; border:1px solid #232a26; border-radius:8px; padding:12px; }}
  .label{{ color:#8a948d; font-size:11px; text-transform:uppercase; }}
  .value{{ margin-top:4px; font-size:20px; font-weight:700; }}
  .table-wrap{{ overflow-x:auto; }}
  table{{ width:100%; border-collapse:collapse; font-size:14px; }}
  th{{ text-align:left; font-size:11px; text-transform:uppercase; color:#8a948d; padding:10px; border-bottom:1px solid #232a26; }}
  footer{{ margin-top:30px; color:#8a948d; font-size:11.5px; line-height:1.6; }}
</style>
</head>
<body>
  <h1>Signal Desk — Daily Results</h1>
  <div class="sub">Generated: {checked_at} · Run type: <b>{run_type}</b> · Latest market data: <b>{latest_data_date or 'unavailable'}</b></div>
  <div class="banner">{banner_text}</div>

  <div class="stats">
  <div class="card"><div class="label">Stocks in watchlist</div><div class="value">{stats['watchlist']}</div></div>

  <div class="card"><div class="label">Passed fundamentals</div><div class="value">{stats['fundamentals_passed']}</div></div>
  <div class="card"><div class="label">Rejected fundamentals</div><div class="value">{stats['fundamentals_rejected']}</div></div>
  <div class="card"><div class="label">Fundamentals unavailable</div><div class="value">{stats['fundamentals_unavailable']}</div></div>

  <div class="card"><div class="label">Price data scanned</div><div class="value">{stats['price_scanned']}</div></div>
  <div class="card"><div class="label">No usable data</div><div class="value">{stats['no_data']}</div></div>

  <div class="card"><div class="label">New golden crosses</div><div class="value">{stats['golden_crosses']}</div></div>
  <div class="card"><div class="label">Already in uptrend</div><div class="value">{stats['wait_uptrend']}</div></div>
  <div class="card"><div class="label">Buy signals</div><div class="value">{stats['buy']}</div></div>
</div>

  <div class="sub">S&amp;P 500 trend: <b>{'UPTREND' if market_up else 'DOWNTREND'}</b></div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Ticker</th><th>Price</th><th>Status</th><th>Entry price</th><th>Stop-loss (-10%)</th><th>Take-profit target (+20%)</th></tr></thead>
      <tbody>{row_html}</tbody>
    </table>
  </div>
  <footer>
    Scheduled scans run at 9:45 PM UTC, which is 12:45 AM Saudi time on the following calendar day.
    During US daylight-saving time this is 5:45 PM New York; during standard time it is 4:45 PM New York.
    Manual workflow runs can appear at any time and are labeled above.
  </footer>
</body>
</html>"""

def main():
    market_up = get_market_uptrend()
    rows = []

    stats = {
        "watchlist": len(WATCHLIST),
        "fundamentals_passed": 0,
        "fundamentals_rejected": 0,
        "fundamentals_unavailable": 0,
        "price_scanned": 0,
        "golden_crosses": 0,
        "buy": 0,
        "wait_uptrend": 0,
        "avoid": 0,
        "no_data": 0,
    }

    latest_dates = []

    for ticker in WATCHLIST:
        fund_ok = passes_fundamental_filter(ticker)

        if fund_ok is False:
            stats["fundamentals_rejected"] += 1
            rows.append({"ticker": ticker, "price": None, "status": "skipped_fundamentals"})
            continue

        if fund_ok is None:
            stats["fundamentals_unavailable"] += 1
        else:
            stats["fundamentals_passed"] += 1

        result = check_ticker(ticker)
        if result is None:
            stats["no_data"] += 1
            rows.append({"ticker": ticker, "price": None, "status": "no_data"})
            continue

        stats["price_scanned"] += 1
        latest_dates.append(result["latest_data_date"])

        if result["crossed_today"]:
            stats["golden_crosses"] += 1

        buy_signal = result["crossed_today"] and market_up
        if buy_signal:
            status = "buy"
        elif result["in_uptrend"]:
            status = "wait_uptrend"
        else:
            status = "avoid"

        stats[status] += 1
        rows.append({"ticker": ticker, "price": result["price"], "status": status})

    checked_at_utc = datetime.now(timezone.utc)
    checked_at_saudi = checked_at_utc + SAUDI_OFFSET
    checked_at = checked_at_saudi.strftime("%Y-%m-%d %I:%M %p") + " Saudi time"

    event_name = os.environ.get("GITHUB_EVENT_NAME", "local")
    run_type = {
        "schedule": "Scheduled daily scan",
        "workflow_dispatch": "Manual workflow run",
        "local": "Local run",
    }.get(event_name, event_name)

    latest_data_date = max(latest_dates) if latest_dates else None

    html = build_html(
        market_up=market_up,
        rows=rows,
        checked_at=checked_at,
        run_type=run_type,
        stats=stats,
        latest_data_date=latest_data_date,
    )

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Wrote docs/index.html")
    print(f"Scan statistics: {stats}")
    print(f"Run type: {run_type}")
    print(f"Latest market data date: {latest_data_date}")

    buy_rows = [row for row in rows if row["status"] == "buy"]
    if buy_rows:
        message = build_telegram_message(buy_rows, checked_at)
        send_telegram_message(message)
    else:
        print("No BUY signals found. No Telegram notification sent.")


if __name__ == "__main__":
    main()
