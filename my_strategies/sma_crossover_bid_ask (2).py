"""
Index Options Backtester + Optimizer (OpenAlgo) — configured to:
1) Use OpenAlgo for historical 1m option data
2) Run for underlyings: NIFTY and SENSEX
3) Save outputs as CSVs
4) Optimization objective: maximize WIN RATE

USAGE
- Set environment variables OPENALGO_API_KEY and OPENALGO_API_HOST.
- Install dependencies: pip install openalgo sqlalchemy pandas numpy plotly
- Run: python backtest_optimizer_winrate.py

OUTPUTS
- CSV: backtest_summary.csv (instrument + day metrics)
- CSV: optimizer_results.csv (grid results sorted by win_rate)
- SQLite DB: option_backtest.db (raw records)

NOTES
- This script only performs a dry-run backtest (no live orders).
- It requires your OpenAlgo instance to provide historical 1m option data via client.history().
- If OpenAlgo does not provide option historical OHLCV, the script will fail to fetch and you should provide a data source or enable myntapi integration.

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
API_KEY = os.getenv('OPENALGO_API_KEY', '74f041f84571131d5bf0ae7bff3ab47802c172c4e34460f40a12cc5fe04e6fc2')
API_HOST = os.getenv('OPENALGO_API_HOST', 'http://127.0.0.1:5000')
client = api(api_key=API_KEY, host=API_HOST)

INDEX_UNDERLYINGS = ['NIFTY', 'SENSEX']
EXCHANGE_INDEX = {'NIFTY':'NSE_INDEX', 'SENSEX':'BSE_INDEX'}
INTERVAL = '1m'
MAX_DAYS = 7
STRIKE_INTERVALS = {'NIFTY':50, 'SENSEX':100}  # adjust if needed

# Strategy defaults
DEFAULTS = {
    'VWMA_PERIOD': 20,
    'ATR_PERIOD': 14,
    'ATR_ROLL_WINDOW': 10,
    'ATR_MULTIPLIER': 1.5,
    'BUY_SELL_RATIO': 1.2,
    'TRAIL_STEP_PROFIT': 1000,
    'TRAIL_STEP_RAISE': 500
}

# Optimizer grid (small; expand if needed)
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

# --- Exchange maps ---
UNDERLYING_EXCHANGE = {
    'NIFTY': 'NSE_INDEX',
    'SENSEX': 'BSE_INDEX'
}

OPTION_EXCHANGE = {
    'NIFTY': 'NFO',
    'SENSEX': 'BFO'
}


def discover_expiries(symbol_base, option_exchange):
    """
    Query the provider's expiry API and return list of datetime.date objects.
    This uses the exact expiry endpoint from your OpenAlgo instance (no fallback to Fridays).
    Raises RuntimeError if expiries cannot be discovered.
    """
    import pandas as _pd
    from datetime import datetime as _datetime

    expiries = []
    try:
        raw = None
        # Try several common client methods (adapt if your client uses different names)
        if hasattr(client, 'expiry'):
            raw = client.expiry(symbol=symbol_base, exchange=option_exchange, instrumenttype='options')
        elif hasattr(client, 'get_expiries'):
            raw = client.get_expiries(symbol=symbol_base, exchange=option_exchange)
        elif hasattr(client, 'expiries'):
            raw = client.expiries(symbol=symbol_base, exchange=option_exchange)
        else:
            raise RuntimeError("No expiry API method found on client.")

        # raw may be a dict like {'data': [...]} or a list
        if isinstance(raw, dict) and 'data' in raw:
            raw_list = raw['data']
        else:
            raw_list = raw

        if not raw_list:
            raise RuntimeError(f"No expiries returned from expiry API for {symbol_base}/{option_exchange}: {raw}")

        parsed = []
        for item in raw_list:
            s = str(item).strip().upper()
            # Try format e.g. 30DEC25
            try:
                dt = _datetime.strptime(s, '%d%b%y').date()
            except Exception:
                # Try ISO format fallback
                try:
                    dt = _pd.to_datetime(s).date()
                except Exception:
                    # skip unknown token
                    continue
            parsed.append(dt)

        if not parsed:
            raise RuntimeError(f"Expiry API returned items but none parsed for {symbol_base}: {raw_list}")

        parsed = sorted(list(set(parsed)))
        return parsed

    except Exception as e:
        raise RuntimeError(f"Failed to discover expiries for {symbol_base} on {option_exchange}: {e}")

#except Exception:
#        pass
    # Try myntapi if available (best-effort)
#    try:
#        import myntapi
#        mynt = myntapi.Mynt()
#        chain = mynt.get_option_chain(symbol_base)
#        if chain and 'expiries' in chain:
#            expiries = [pd.to_datetime(x).date() for x in chain['expiries']]
#            expiries = sorted([d for d in expiries if d >= datetime.now().date()])
#            if expiries:
#                return expiries
#    except Exception:
#        pass
#    # fallback: next Fridays
#    today = datetime.now().date()
#    for d in range(0, MAX_DAYS + 14):
#        candidate = today + timedelta(days=d)
#        if candidate.weekday() == 4:
#            expiries.append(candidate)
#        if len(expiries) >= 3:
#            break
#    return expiries


def round_to_strike(price, strike_interval):
    return int(round(price / strike_interval) * strike_interval)


def build_option_symbol(base, expiry_date, strike, opt_type, exchange_hint='NSE'):
    dd = expiry_date.day
    mon = expiry_date.strftime('%b').upper()
    yy = expiry_date.strftime('%y')
    ot = str(opt_type).upper()
    if ot in ('CE','C'):
        ochar = 'C'
    elif ot in ('PE','P'):
        ochar = 'P'
    else:
        ochar = ot[0] if ot else 'P'
    strike_str = str(int(strike))
    if exchange_hint and exchange_hint.upper().startswith('BSE'):
        cepe = 'CE' if ochar=='C' else 'PE'
        return f"{base}{dd:02d}{mon}{yy}O{strike_str}{cepe}"
    else:
        return f"{base}{dd:02d}{mon}{yy}{ochar}{strike_str}"


def select_strikes(open_price, strike_interval, dte):
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
    try:
        df = client.history(symbol=symbol, exchange=exchange, interval=interval,
                            start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

# compute metrics

def compute_trade_metrics(trades):
    entries = [t for t in trades if t['type']=='ENTRY']
    exits = [t for t in trades if t['type']=='EXIT']
    num_trades = min(len(entries), len(exits))
    if num_trades == 0:
        return {'trades':0,'wins':0,'losses':0,'win_rate':0.0,'net_pnl':0.0,'avg_pnl':0.0,'max_drawdown':0.0,'profit_factor':None}
    pnls = [e['pnl'] for e in exits]
    wins = sum(1 for p in pnls if p>0)
    losses = sum(1 for p in pnls if p<=0)
    net = sum(pnls)
    avg = np.mean(pnls) if pnls else 0
    win_rate = wins/num_trades if num_trades>0 else 0
    eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq)
    max_dd = dd.max() if len(dd)>0 else 0
    profit_factor = sum([p for p in pnls if p>0]) / abs(sum([p for p in pnls if p<0])) if any(p<0 for p in pnls) else None
    return {'trades':num_trades,'wins':wins,'losses':losses,'win_rate':win_rate,'net_pnl':net,'avg_pnl':avg,'max_drawdown':max_dd,'profit_factor':profit_factor}

# backtest single option per day

def backtest_option_day(option_symbol, option_exchange, day_date, params, capital=10000):
    start_dt = datetime.combine(day_date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)
    df = fetch_option_history(option_symbol, option_exchange, '1m', start_dt, end_dt)
    if df is None or df.empty:
        return [], None
    df['mid'] = (df['high'] + df['low'])/2
    df['volume'] = df['volume'].fillna(0)
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
        buy_vol = 0.55 * row['volume'] if row['close'] >= row['open'] else 0.45 * row['volume']
        sell_vol = row['volume'] - buy_vol
        atr_roll_mean = df['ATR'].iloc[max(0, i-params['ATR_ROLL_WINDOW']):i].mean()
        atr_now = row['ATR']
        if position is None:
            if (buy_vol > sell_vol * params['BUY_SELL_RATIO']) and (row['VWMA'] is not None and row['close'] > row['VWMA']) and (atr_roll_mean>0 and atr_now > atr_roll_mean * params['ATR_MULTIPLIER']):
                position = 'LONG'
                entry_price = row['close']
                qty = math.floor(capital / entry_price) if entry_price>0 else 0
                highest = entry_price
                trades.append({'type':'ENTRY','time':ts,'price':entry_price,'qty':qty})
        else:
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
    metrics = compute_trade_metrics(trades)
    return trades, metrics

# run backtests across underlyings, days, strikes

def run_backtests(params):
    results = []
    today = datetime.now().date()
    days = [today - timedelta(days=d) for d in range(0, MAX_DAYS)]
    days = sorted(days)
    for underlying in INDEX_UNDERLYINGS:
        exchange = OPTION_EXCHANGE.get(underlying)
        expiries = discover_expiries(underlying, exchange)
        if not expiries:
            continue
        for day in days:
            # fetch underlying open
            try:
                df_under = fetch_option_history(underlying, exchange, INTERVAL, day, day+timedelta(days=1))
                if df_under is None or df_under.empty:
                    continue
                open_price = float(df_under['open'].iloc[0])
            except Exception:
                continue
            nearest_exp = expiries[0]
            dte = (nearest_exp - day).days
            strikes = select_strikes(open_price, STRIKE_INTERVALS.get(underlying,50), dte)
            option_symbols = []
            for s in strikes:
                # CE and PE symbols
                opt_ce = build_option_symbol(underlying, nearest_exp, s, 'CE', exchange_hint=exchange)
                opt_pe = build_option_symbol(underlying, nearest_exp, s, 'PE', exchange_hint=exchange)
                option_symbols.extend([(opt_ce, 'NFO'), (opt_pe, 'NFO')])
            for opt_sym, opt_ex in option_symbols:
                trades, metrics = backtest_option_day(opt_sym, opt_ex, day, params)
                if metrics:
                    res = {'underlying':underlying,'day':day,'symbol':opt_sym}
                    res.update(metrics)
                    results.append(res)
    df_res = pd.DataFrame(results)
    return df_res

# optimizer targeting win_rate

def optimizer(target='win_rate'):
    grid_keys = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in grid_keys]))
    all_results = []
    for combo in combos:
        params = DEFAULTS.copy()
        for k, val in zip(grid_keys, combo):
            params[k] = val
        print('Running grid item:', params)
        df_res = run_backtests(params)
        if df_res.empty:
            score = 0
        else:
            # aggregate win_rate across instruments/days as mean
            score = df_res['win_rate'].mean()
        all_results.append({'params':params,'score':score})
        print('Score (mean win_rate):', score)
    # save results
    rows = []
    for r in all_results:
        row = {**r['params']}
        row['score'] = r['score']
        rows.append(row)
    df_all = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'optimizer_results.csv')
    df_all.sort_values('score', ascending=False).to_csv(out_csv, index=False)
    print('Optimizer finished. Results saved to', out_csv)
    return df_all

# ---------------- main ----------------
if __name__ == '__main__':
    print('Starting optimizer (objective = WIN RATE) for NIFTY and SENSEX...')
    df_optimizer = optimizer()
    print('Top 3 parameter sets by win_rate:')
    print(df_optimizer.sort_values('score', ascending=False).head(3))
    # run final backtest with best params and save summary
    best_params = df_optimizer.sort_values('score', ascending=False).iloc[0].to_dict()
    # convert best_params back to params dict
    params = DEFAULTS.copy()
    for k in GRID.keys():
        if k in best_params:
            params[k] = best_params[k]
    summary = run_backtests(params)
    out_summary = os.path.join(OUT_DIR, 'backtest_summary.csv')
    summary.to_csv(out_summary, index=False)
    print('Backtest summary saved to', out_summary)
