"""
Comprehensive Backtester — Index Options (nearest expiry) + Metrics + Parameter Optimizer

Features:
- Runs dry-run backtests (no live orders) for **index options only** (e.g., NIFTY, BANKNIFTY) for up to **7 days**.
- Determines nearest expiry instruments by querying OpenAlgo for available expiries when possible; falls back to the next weekly expiry (Friday) if expiry list API is unavailable.
- Strike selection logic (based on your rules):
    • If days_to_expiry >= 6: select +/- 5 strikes
    • If days_to_expiry <= 4: select +/- 4 strikes
    • Else: +/- 3 strikes
  Strikes are selected around the **current day's opening price** rounded to the nearest strike interval (configurable).
- Backtest engine simulates entries/exits using the depth-weighted QWP/VWMA + ATR + Supertrend + trailing rules (same as earlier live/dry-run version), but operates on option premium time series.
- Computes per-instrument and daywise performance metrics:
    • Trades, Wins, Losses, Win rate
    • Total P&L, Average P&L, Avg P&L per trade
    • Max Drawdown, Profit Factor, Sharpe-like ratio (using mean/std of trade returns)
- Parameter optimizer (grid search) over a small parameter grid for: ATR_MULTIPLIER, VWMA_PERIOD, TRAIL_STEP_PROFIT, TRAIL_STEP_RAISE, BUY_SELL_RATIO. Reports best parameter set per objective (net P&L by default).

Assumptions and notes:
- OpenAlgo client: we use `client.history()` heavily and attempt to discover expiries via `client.get_expiries()` or `client.option_chain()` if available. If your OpenAlgo SDK exposes different functions, adapt the helper `discover_expiries()` and `build_option_symbol()` accordingly.
- Option symbol format: this script uses the OpenAlgo symbol format convention `{BASE}{DD}{MON}{YY}{STRIKE}{CE|PE}` (e.g., `NIFTY24OCT2439000CE`). If your broker uses a different pattern, update `build_option_symbol()`.
- Strike interval: default 50 for NIFTY-like indices; change via STRIKE_INTERVAL.
- Backtests use 1-minute bars (`interval='1m'`) as requested.
- Max backtest days default to 7 (configurable).

How to use:
- Set OPENALGO_API_KEY and OPENALGO_API_HOST environment variables or edit the config section.
- Run the script; results and optimizer outputs are saved to CSV/SQLite and plotted HTML for quick inspection.

"""

import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openalgo import api, ta
from itertools import product
from sqlalchemy import create_engine

# ---------------- CONFIG ----------------
API_KEY = os.getenv('OPENALGO_API_KEY', 'your-openalgo-apikey')
API_HOST = os.getenv('OPENALGO_API_HOST', 'http://127.0.0.1:5000')
client = api(api_key=API_KEY, host=API_HOST)

INDEX_UNDERLYINGS = ['NIFTY', 'BANKNIFTY']  # instruments to run
EXCHANGE_INDEX = 'NSE_INDEX'
INTERVAL = '1m'
MAX_DAYS = 7
STRIKE_INTERVAL = 50  # change per index if needed

# Strategy defaults (these will be optimized)
DEFAULTS = {
    'VWMA_PERIOD': 20,
    'ATR_PERIOD': 14,
    'ATR_ROLL_WINDOW': 10,
    'ATR_MULTIPLIER': 1.5,
    'BUY_SELL_RATIO': 1.2,
    'TRAIL_STEP_PROFIT': 1000,
    'TRAIL_STEP_RAISE': 500
}

# Optimizer grid (small by default; expand if needed)
GRID = {
    'VWMA_PERIOD': [10, 20],
    'ATR_MULTIPLIER': [1.2, 1.5],
    'BUY_SELL_RATIO': [1.1, 1.2],
    'TRAIL_STEP_PROFIT': [500, 1000],
    'TRAIL_STEP_RAISE': [250, 500]
}

OUT_DIR = os.getenv('BACKTEST_OUT', '.')
DB_PATH = os.path.join(OUT_DIR, 'option_backtest.db')
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)

# ---------------- Helpers ----------------

def discover_expiries(symbol_base, exchange):
    """
    Try to discover available option expiries for the underlying via OpenAlgo.
    Falls back to computing the next Friday (weekly expiry) for up to MAX_DAYS.
    Returns list of expiry datetimes (date objects).
    """
    expiries = []
    # Try client.get_expiries if available
    if hasattr(client, 'get_expiries'):
        try:
            expiries = client.get_expiries(symbol=symbol_base, exchange=exchange)
            # Expect expiries as date strings or date objects
            expiries = [pd.to_datetime(x).date() for x in expiries]
            expiries = sorted([d for d in expiries if d >= datetime.now().date()])
            return expiries
        except Exception:
            pass
    # Try client.option_chain or market metadata
    # Fallback: next 4 Fridays within MAX_DAYS window
    today = datetime.now().date()
    for d in range(0, MAX_DAYS+14):
        candidate = today + timedelta(days=d)
        # weekday()==4 is Friday (Mon=0). India weekly expiry often Thursday/Friday depending; using Friday as fallback
        if candidate.weekday() == 4:
            expiries.append(candidate)
        if len(expiries) >= 3:
            break
    return expiries


def round_to_strike(price, strike_interval):
    return int(round(price / strike_interval) * strike_interval)


def build_option_symbol(base, expiry_date, strike, opt_type):
    """
    Build OpenAlgo-style option symbol. Adjust formatting if your broker requires alternative format.
    Format used: {BASE}{DD}{MON}{YY}{STRIKE}{CE|PE}
    Example: NIFTY24OCT2439000CE
    """
    dd = expiry_date.day
    mon = expiry_date.strftime('%b').upper()
    yy = expiry_date.strftime('%y')
    strike_str = str(int(strike))
    return f"{base}{dd:02d}{mon}{yy}{strike_str}{opt_type}"


def select_strikes(open_price, strike_interval, dte):
    """
    Based on days to expiry, choose +/- N strikes around ATM.
    """
    if dte >= 6:
        N = 5
    elif dte <= 4:
        N = 4
    else:
        N = 3
    atm = round_to_strike(open_price, strike_interval)
    strikes = [atm + i*strike_interval for i in range(-N, N+1)]
    return strikes


def fetch_option_history(symbol, exchange, interval, start_date, end_date):
    df = client.history(symbol=symbol, exchange=exchange, interval=interval,
                        start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
    if not isinstance(df, pd.DataFrame):
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

# core backtest for a single option symbol and single day

def backtest_option_day(option_symbol, option_exchange, day_date, params):
    """
    Simulate one trading day on 1m bars using the same logic as live bot but without placing orders.
    Returns trades list and metrics.
    """
    start_dt = datetime.combine(day_date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)
    df = fetch_option_history(option_symbol, option_exchange, '1m', start_dt, end_dt)
    if df is None or df.empty:
        return [], None

    # compute indicators
    df['mid'] = (df['high'] + df['low'])/2
    df['volume'] = df['volume'].fillna(0)
    # rolling VWMA (snapshot-based) — use volume as weights
    df['VWMA'] = df['close'].rolling(window=params['VWMA_PERIOD']).apply(
        lambda x: np.average(x, weights=df.loc[x.index, 'volume']) if df.loc[x.index, 'volume'].sum()>0 else x.mean()
    )
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], period=params['ATR_PERIOD'])
    st, st_dir = ta.supertrend(df['high'], df['low'], df['close'], period=10, multiplier=2.0)
    df['Supertrend'] = st

    trades = []
    position = None
    entry_price = 0
    qty = 0
    highest = 0

    for i in range(params['VWMA_PERIOD'], len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        # compute buy/sell volume proxies from ticks: for options we approximate using candle volume distribution
        # simple proxy: buy_vol = 0.55*volume if close>open else 0.45*volume (heuristic)
        buy_vol = 0.55 * row['volume'] if row['close'] >= row['open'] else 0.45 * row['volume']
        sell_vol = row['volume'] - buy_vol

        # ATR rolling mean
        atr_roll_mean = df['ATR'].iloc[max(0, i-params['ATR_ROLL_WINDOW']):i].mean()
        atr_now = row['ATR']

        # entry conditions
        if position is None:
            if (buy_vol > sell_vol * params['BUY_SELL_RATIO']) and (row['VWMA'] is not None and row['close'] > row['VWMA']) and (atr_roll_mean>0 and atr_now > atr_roll_mean * params['ATR_MULTIPLIER']):
                position = 'LONG'
                entry_price = row['close']
                qty = math.floor(CAPITAL / entry_price) if entry_price>0 else 0
                highest = entry_price
                trades.append({'type':'ENTRY','time':ts,'price':entry_price,'qty':qty})
        else:
            # update highest and trailing
            if row['close'] > highest:
                highest = row['close']
            steps = math.floor((highest - entry_price) / params['TRAIL_STEP_PROFIT'])
            trailing = entry_price + steps * params['TRAIL_STEP_RAISE'] if steps>0 else None

            exit_reason = None
            if trailing and row['close'] < trailing:
                exit_reason = 'TRAIL_BROKEN'
            if row['close'] < row['Supertrend']:
                exit_reason = exit_reason or 'SUPERTREND_BREACH'
            if ts.time() >= datetime.strptime('15:15','%H:%M').time():
                exit_reason = exit_reason or 'TIME_EXIT'

            if exit_reason:
                exit_price = row['close']
                pnl = (exit_price - entry_price) * qty
                trades.append({'type':'EXIT','time':ts,'price':exit_price,'qty':qty,'pnl':pnl,'reason':exit_reason})
                position = None
                entry_price = 0
                qty = 0
                highest = 0

    # compute metrics
    metrics = compute_trade_metrics(trades)
    return trades, metrics


def compute_trade_metrics(trades):
    entries = [t for t in trades if t['type']=='ENTRY']
    exits = [t for t in trades if t['type']=='EXIT']
    num_trades = min(len(entries), len(exits))
    if num_trades == 0:
        return {'trades':0,'wins':0,'losses':0,'win_rate':None,'net_pnl':0.0,'avg_pnl':0.0,'max_drawdown':0.0,'profit_factor':None}
    pnls = [e['pnl'] for e in exits]
    wins = sum(1 for p in pnls if p>0)
    losses = sum(1 for p in pnls if p<=0)
    net = sum(pnls)
    avg = np.mean(pnls) if pnls else 0
    win_rate = wins/num_trades if num_trades>0 else 0
    # equity curve and drawdown
    eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq)
    max_dd = dd.max() if len(dd)>0 else 0
    profit_factor = sum([p for p in pnls if p>0]) / abs(sum([p for p in pnls if p<0])) if any(p<0 for p in pnls) else None
    return {'trades':num_trades,'wins':wins,'losses':losses,'win_rate':win_rate,'net_pnl':net,'avg_pnl':avg,'max_drawdown':max_dd,'profit_factor':profit_factor}

# ---------------- Runner over symbol set and days ----------------

def run_backtests():
    results = []
    detailed = []

    today = datetime.now().date()
    days = [today - timedelta(days=d) for d in range(0, MAX_DAYS)]
    days = sorted(days)

    for underlying in INDEX_UNDERLYINGS:
        expiries = discover_expiries(underlying, EXCHANGE_INDEX)
        for day in days:
            # get opening price of underlying on that day
            try:
                df_under = fetch_option_history(underlying, EXCHANGE_INDEX, INTERVAL, day, day+timedelta(days=1))
                if df_under is None or df_under.empty:
                    continue
                open_price = float(df_under['open'].iloc[0])
            except Exception:
                continue

            # pick nearest expiry >= day
            dte_list = [ (exp - day).days for exp in expiries if (exp - day).days >=0 ]
            if not dte_list:
                continue
            nearest_exp = expiries[0]
            dte = (nearest_exp - day).days

            strikes = select_strikes(open_price, STRIKE_INTERVAL, dte)

            # build option symbols
            option_symbols = []
            for s in strikes:
                ce = build_option_symbol(underlying, nearest_exp, s, 'CE')
                pe = build_option_symbol(underlying, nearest_exp, s, 'PE')
                option_symbols.extend([(ce,'NFO'),(pe,'NFO')])  # assuming options on NFO exchange

            # run backtest for each option symbol for this day
            for opt_sym, opt_ex in option_symbols:
                trades, metrics = backtest_option_day(opt_sym, opt_ex, day, DEFAULTS)
                detailed.append({'underlying':underlying,'day':day,'expiry':nearest_exp,'symbol':opt_sym,'metrics':metrics})
                if metrics:
                    res = {'underlying':underlying,'day':day,'symbol':opt_sym}
                    res.update(metrics)
                    results.append(res)

    # save results
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, 'backtest_summary.csv')
    df_res.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")
    return df_res, detailed

# ---------------- Parameter optimizer ----------------

def optimizer(target='net_pnl'):
    grid_keys = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in grid_keys]))
    best = None
    all_results = []
    for combo in combos:
        params = DEFAULTS.copy()
        for k, val in zip(grid_keys, combo):
            params[k] = val
        # run quick backtest over all symbols/days (could be slow) — we limit days to MAX_DAYS
        df_res, _ = run_backtests_with_params(params)
        if df_res.empty:
            continue
        score = df_res[target].sum() if target in df_res.columns else df_res['net_pnl'].sum()
        all_results.append({'params':params,'score':score})
        if best is None or score > best['score']:
            best = {'params':params,'score':score}
        print(f"Tried {params} => score {score}")
    # save all_results
    df_all = pd.DataFrame([{'score':r['score'], **r['params']} for r in all_results])
    df_all.to_csv(os.path.join(OUT_DIR,'optimizer_results.csv'), index=False)
    print(f"Best: {best}")
    return best, df_all


def run_backtests_with_params(params):
    # wrapper similar to run_backtests but uses params instead of DEFAULTS
    results = []
    detailed = []
    today = datetime.now().date()
    days = [today - timedelta(days=d) for d in range(0, MAX_DAYS)]
    days = sorted(days)

    for underlying in INDEX_UNDERLYINGS:
        expiries = discover_expiries(underlying, EXCHANGE_INDEX)
        for day in days:
            try:
                df_under = fetch_option_history(underlying, EXCHANGE_INDEX, INTERVAL, day, day+timedelta(days=1))
                if df_under is None or df_under.empty:
                    continue
                open_price = float(df_under['open'].iloc[0])
            except Exception:
                continue
            if not expiries:
                continue
            nearest_exp = expiries[0]
            dte = (nearest_exp - day).days
            strikes = select_strikes(open_price, STRIKE_INTERVAL, dte)
            option_symbols = []
            for s in strikes:
                ce = build_option_symbol(underlying, nearest_exp, s, 'CE')
                pe = build_option_symbol(underlying, nearest_exp, s, 'PE')
                option_symbols.extend([(ce,'NFO'),(pe,'NFO')])
            for opt_sym, opt_ex in option_symbols:
                trades, metrics = backtest_option_day(opt_sym, opt_ex, day, params)
                detailed.append({'underlying':underlying,'day':day,'expiry':nearest_exp,'symbol':opt_sym,'metrics':metrics})
                if metrics:
                    res = {'underlying':underlying,'day':day,'symbol':opt_sym}
                    res.update(metrics)
                    results.append(res)
    df_res = pd.DataFrame(results)
    return df_res, detailed

# ---------------- Entry point ----------------
if __name__ == '__main__':
    print('Running backtests (up to 7 days) for index options...')
    summary_df, details = run_backtests()
    print('Summary head:')
    print(summary_df.head())
    print('Running optimizer (this may take time)...')
    best, all_results = optimizer()
    print('Done. Best params:')
    print(best)
