"""
Part A: Data preparation
Loads both datasets, documents shape/missing/dupes, aligns by date,
and builds account-day and day-level metrics.
"""
import pandas as pd
import numpy as np

RAW_HIST = '/mnt/user-data/uploads/historical_data.csv'
RAW_FG = '/mnt/user-data/uploads/fear_greed_index.csv'
OUT = '/home/claude/project/data'

# ---------- Load ----------
hist = pd.read_csv(RAW_HIST)
fg = pd.read_csv(RAW_FG)

report = []
report.append("=== RAW SHAPES ===")
report.append(f"historical_data.csv: {hist.shape[0]:,} rows x {hist.shape[1]} cols")
report.append(f"fear_greed_index.csv: {fg.shape[0]:,} rows x {fg.shape[1]} cols")

report.append("\n=== MISSING VALUES ===")
report.append(f"historical_data.csv missing per col:\n{hist.isna().sum()[hist.isna().sum()>0]}")
report.append(f"fear_greed_index.csv missing per col:\n{fg.isna().sum()[fg.isna().sum()>0]}")
if hist.isna().sum().sum() == 0:
    report.append("historical_data.csv: no missing values")
if fg.isna().sum().sum() == 0:
    report.append("fear_greed_index.csv: no missing values")

report.append("\n=== DUPLICATES ===")
report.append(f"historical_data.csv duplicate rows: {hist.duplicated().sum()}")
report.append(f"fear_greed_index.csv duplicate rows: {fg.duplicated().sum()}")

# ---------- Clean / typing ----------
hist['dt'] = pd.to_datetime(hist['Timestamp IST'], format='%d-%m-%Y %H:%M')
hist['date'] = hist['dt'].dt.date
fg['date'] = pd.to_datetime(fg['date']).dt.date

# Collapse 5-way sentiment into simpler Fear/Greed/Neutral buckets, keep original too
sent_map = {
    'Extreme Fear': 'Fear', 'Fear': 'Fear',
    'Neutral': 'Neutral',
    'Greed': 'Greed', 'Extreme Greed': 'Greed'
}
fg['sentiment_simple'] = fg['classification'].map(sent_map)

report.append("\n=== DATE COVERAGE ===")
report.append(f"historical_data.csv date range: {hist['date'].min()} to {hist['date'].max()}")
report.append(f"fear_greed_index.csv date range: {fg['date'].min()} to {fg['date'].max()}")

# Note on 'leverage' field: not present in this Hyperliquid export (no margin/equity
# column to derive it). We proxy trade intensity/risk with Size USD (notional) instead
# and flag this as a limitation in the README.
report.append("\n=== NOTE ===")
report.append("No 'leverage' column exists in historical_data.csv (no equity/margin field "
               "to derive it). Position notional (Size USD) is used as the risk/intensity proxy instead.")

# A trade is a "close" (realized PnL event) if Direction contains Close, or is a
# flat Buy/Sell/reversal on spot-like symbols where Closed PnL is populated.
close_directions = ['Close Long', 'Close Short', 'Short > Long', 'Long > Short',
                     'Buy', 'Sell', 'Liquidated Isolated Short', 'Settlement']
hist['is_close_event'] = hist['Direction'].isin(close_directions) & (hist['Closed PnL'] != 0)
# more permissive: any row where Closed PnL != 0 is a realized PnL event
hist['is_realized'] = hist['Closed PnL'] != 0

# ---------- Merge sentiment onto trades ----------
trades = hist.merge(fg[['date', 'classification', 'sentiment_simple', 'value']],
                     on='date', how='left')
missing_sent = trades['sentiment_simple'].isna().sum()
report.append(f"\nTrades with no matching sentiment day: {missing_sent:,} "
              f"({missing_sent/len(trades)*100:.2f}%)")
trades = trades.dropna(subset=['sentiment_simple']).copy()

trades.to_csv(f'{OUT}/trades_with_sentiment.csv', index=False)

# ---------- Account-day metrics ----------
grp = trades.groupby(['Account', 'date'])

acct_day = grp.agg(
    n_trades=('Trade ID', 'count'),
    volume_usd=('Size USD', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    realized_pnl=('Closed PnL', 'sum'),
    fees=('Fee', 'sum'),
    n_buy=('Side', lambda s: (s == 'BUY').sum()),
    n_sell=('Side', lambda s: (s == 'SELL').sum()),
).reset_index()

# win rate: share of realized-PnL trades that were profitable
realized = trades[trades['is_realized']]
win = realized.groupby(['Account', 'date']).apply(
    lambda d: pd.Series({
        'n_realized_trades': len(d),
        'n_wins': (d['Closed PnL'] > 0).sum(),
    }), include_groups=False
).reset_index()
win['win_rate'] = np.where(win['n_realized_trades'] > 0,
                            win['n_wins'] / win['n_realized_trades'], np.nan)

acct_day = acct_day.merge(win[['Account', 'date', 'n_realized_trades', 'win_rate']],
                           on=['Account', 'date'], how='left')

acct_day['long_short_ratio'] = np.where(acct_day['n_sell'] > 0,
                                         acct_day['n_buy'] / acct_day['n_sell'], np.nan)
acct_day['net_pnl_after_fees'] = acct_day['realized_pnl'] - acct_day['fees']

# attach sentiment
day_sent = fg[['date', 'classification', 'sentiment_simple', 'value']].drop_duplicates('date')
acct_day = acct_day.merge(day_sent, on='date', how='left')
acct_day.to_csv(f'{OUT}/account_day_metrics.csv', index=False)

report.append(f"\n=== ACCOUNT-DAY TABLE ===")
report.append(f"Rows: {len(acct_day):,} (unique account-day combos)")
report.append(f"Unique accounts: {acct_day['Account'].nunique()}")
report.append(f"Date span used (post-merge): {acct_day['date'].min()} to {acct_day['date'].max()}")

# ---------- Day-level (market-wide) metrics ----------
day = trades.groupby('date').agg(
    n_trades=('Trade ID', 'count'),
    n_active_accounts=('Account', 'nunique'),
    total_volume_usd=('Size USD', 'sum'),
    total_realized_pnl=('Closed PnL', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    n_buy=('Side', lambda s: (s == 'BUY').sum()),
    n_sell=('Side', lambda s: (s == 'SELL').sum()),
).reset_index()
day['long_short_ratio'] = day['n_buy'] / day['n_sell']
day = day.merge(day_sent, on='date', how='left')

day_realized = realized.groupby('date').apply(
    lambda d: pd.Series({'win_rate': (d['Closed PnL'] > 0).mean()}), include_groups=False
).reset_index()
day = day.merge(day_realized, on='date', how='left')
day.to_csv(f'{OUT}/day_level_metrics.csv', index=False)

report.append(f"\n=== DAY-LEVEL TABLE ===")
report.append(f"Rows: {len(day):,} trading days")
report.append(f"Sentiment split (days): \n{day['sentiment_simple'].value_counts()}")

with open('/home/claude/project/output/01_data_prep_report.txt', 'w') as f:
    f.write('\n'.join(str(x) for x in report))

print('\n'.join(str(x) for x in report))
print("\nSaved: trades_with_sentiment.csv, account_day_metrics.csv, day_level_metrics.csv")
