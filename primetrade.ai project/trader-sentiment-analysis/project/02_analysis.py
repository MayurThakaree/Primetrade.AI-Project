"""
Part B: Analysis
Fear vs Greed performance & behavior, segmentation, insights.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 110

DATA = '/home/claude/project/data'
CHARTS = '/home/claude/project/charts'
OUT = '/home/claude/project/output'

acct_day = pd.read_csv(f'{DATA}/account_day_metrics.csv')
day = pd.read_csv(f'{DATA}/day_level_metrics.csv')
trades = pd.read_csv(f'{DATA}/trades_with_sentiment.csv')

# keep only Fear/Greed for the core comparison (drop Neutral to sharpen contrast),
# but keep a 3-way version too
for df in (acct_day, day):
    df['sentiment_simple'] = pd.Categorical(df['sentiment_simple'],
                                             categories=['Fear', 'Neutral', 'Greed'])

report = []

# ---------------- Q1: performance differs Fear vs Greed? ----------------
perf = acct_day.groupby('sentiment_simple', observed=True).agg(
    avg_daily_pnl=('realized_pnl', 'mean'),
    median_daily_pnl=('realized_pnl', 'median'),
    avg_win_rate=('win_rate', 'mean'),
    pnl_std=('realized_pnl', 'std'),
    n_obs=('realized_pnl', 'count'),
).round(2)
report.append("=== Performance by sentiment (account-day level) ===")
report.append(perf.to_string())

# drawdown proxy: worst daily pnl per account, compared across sentiment
dd = acct_day.groupby('sentiment_simple', observed=True)['realized_pnl'].quantile(0.05).round(2)
report.append("\n5th percentile daily PnL (drawdown proxy) by sentiment:")
report.append(dd.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
order = ['Fear', 'Neutral', 'Greed']
colors = {'Fear': '#c0392b', 'Neutral': '#7f8c8d', 'Greed': '#27ae60'}
bar_data = perf.loc[order, 'avg_daily_pnl']
axes[0].bar(order, bar_data, color=[colors[o] for o in order])
axes[0].set_title('Avg daily realized PnL per account by sentiment')
axes[0].set_ylabel('USD')
axes[0].axhline(0, color='black', lw=0.8)

wr_data = perf.loc[order, 'avg_win_rate']
axes[1].bar(order, wr_data, color=[colors[o] for o in order])
axes[1].set_title('Avg win rate by sentiment')
axes[1].set_ylabel('Win rate')
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{CHARTS}/01_pnl_winrate_by_sentiment.png')
plt.close()

# boxplot of daily pnl distribution (clipped for readability)
fig, ax = plt.subplots(figsize=(7, 4.5))
clip = acct_day['realized_pnl'].clip(-500, 500)
data_box = [clip[acct_day['sentiment_simple'] == s] for s in order]
bp = ax.boxplot(data_box, labels=order, patch_artist=True, showfliers=False)
for patch, o in zip(bp['boxes'], order):
    patch.set_facecolor(colors[o])
    patch.set_alpha(0.6)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Distribution of daily realized PnL per account (clipped to ±$500)')
ax.set_ylabel('USD')
plt.tight_layout()
plt.savefig(f'{CHARTS}/02_pnl_distribution_boxplot.png')
plt.close()

# ---------------- Q2: behavior differs Fear vs Greed? ----------------
behav = acct_day.groupby('sentiment_simple', observed=True).agg(
    avg_trades_per_acct_day=('n_trades', 'mean'),
    avg_trade_size_usd=('avg_trade_size_usd', 'mean'),
    avg_long_short_ratio=('long_short_ratio', 'mean'),
    avg_volume_usd=('volume_usd', 'mean'),
).round(2)
report.append("\n=== Behavior by sentiment (account-day level) ===")
report.append(behav.to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
axes[0].bar(order, behav.loc[order, 'avg_trades_per_acct_day'], color=[colors[o] for o in order])
axes[0].set_title('Avg # trades per account/day')
axes[1].bar(order, behav.loc[order, 'avg_trade_size_usd'], color=[colors[o] for o in order])
axes[1].set_title('Avg trade size (USD)')
axes[2].bar(order, behav.loc[order, 'avg_long_short_ratio'], color=[colors[o] for o in order])
axes[2].axhline(1, color='black', lw=0.8, ls='--')
axes[2].set_title('Avg long/short ratio\n(buy count / sell count)')
plt.tight_layout()
plt.savefig(f'{CHARTS}/03_behavior_by_sentiment.png')
plt.close()

# time series overlay: daily market volume & pnl vs sentiment value
day_sorted = day.sort_values('date')
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ax1.plot(pd.to_datetime(day_sorted['date']), day_sorted['total_realized_pnl'].rolling(7).mean(),
         color='#2c3e50', label='7d avg realized PnL (all accounts)')
ax1.set_ylabel('7d avg total realized PnL (USD)')
ax1.axhline(0, color='grey', lw=0.6)
ax2 = ax1.twinx()
ax2.plot(pd.to_datetime(day_sorted['date']), day_sorted['value'], color='#e67e22', alpha=0.6,
         label='Fear/Greed index value')
ax2.set_ylabel('Fear/Greed index (0-100)')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.88))
ax1.set_title('Market-wide realized PnL vs Fear/Greed index over time')
plt.tight_layout()
plt.savefig(f'{CHARTS}/04_pnl_vs_sentiment_timeseries.png')
plt.close()

# ---------------- Segmentation ----------------
# Segment 1: high vs low position-size ("leverage proxy") traders
acct_summary = acct_day.groupby('Account').agg(
    total_pnl=('realized_pnl', 'sum'),
    avg_trade_size=('avg_trade_size_usd', 'mean'),
    avg_daily_trades=('n_trades', 'mean'),
    active_days=('date', 'nunique'),
    avg_win_rate=('win_rate', 'mean'),
    pnl_std=('realized_pnl', 'std'),
).reset_index()
acct_summary['size_segment'] = pd.qcut(acct_summary['avg_trade_size'], 2, labels=['Low size', 'High size'])
acct_summary['freq_segment'] = pd.qcut(acct_summary['avg_daily_trades'], 2, labels=['Infrequent', 'Frequent'])
# consistency: coefficient of variation of daily pnl (lower = more consistent), among profitable accounts
acct_summary['consistency'] = acct_summary['pnl_std'] / acct_summary['total_pnl'].abs().replace(0, np.nan)
acct_summary['consistency_segment'] = np.where(
    acct_summary['total_pnl'] > 0,
    np.where(acct_summary['consistency'] < acct_summary['consistency'].median(), 'Consistent winner', 'Inconsistent winner'),
    'Net loser'
)
acct_summary.to_csv(f'{DATA}/account_summary_segments.csv', index=False)

report.append("\n=== Segment: trade size (proxy for leverage/risk appetite) ===")
seg_size = acct_day.merge(acct_summary[['Account', 'size_segment']], on='Account').groupby(
    ['size_segment', 'sentiment_simple'], observed=True)['realized_pnl'].mean().unstack().round(2)
report.append(seg_size.to_string())

report.append("\n=== Segment: trading frequency ===")
seg_freq = acct_day.merge(acct_summary[['Account', 'freq_segment']], on='Account').groupby(
    ['freq_segment', 'sentiment_simple'], observed=True)['realized_pnl'].mean().unstack().round(2)
report.append(seg_freq.to_string())

report.append("\n=== Segment: consistency ===")
report.append(acct_summary['consistency_segment'].value_counts().to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
seg_size.T.plot(kind='bar', ax=axes[0], color=['#8e44ad', '#f39c12'])
axes[0].set_title('Avg daily PnL: trade-size segment x sentiment')
axes[0].axhline(0, color='black', lw=0.6)
axes[0].tick_params(axis='x', rotation=0)

seg_freq.T.plot(kind='bar', ax=axes[1], color=['#16a085', '#2980b9'])
axes[1].set_title('Avg daily PnL: frequency segment x sentiment')
axes[1].axhline(0, color='black', lw=0.6)
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(f'{CHARTS}/05_segments_pnl_by_sentiment.png')
plt.close()

# consistency segment pie
fig, ax = plt.subplots(figsize=(5.5, 5))
counts = acct_summary['consistency_segment'].value_counts()
ax.pie(counts, labels=counts.index, autopct='%1.0f%%', colors=['#27ae60', '#f1c40f', '#c0392b'])
ax.set_title('Trader consistency segments (n=%d accounts)' % len(acct_summary))
plt.tight_layout()
plt.savefig(f'{CHARTS}/06_consistency_segments_pie.png')
plt.close()

with open(f'{OUT}/02_analysis_report.txt', 'w') as f:
    f.write('\n'.join(str(x) for x in report))
print('\n'.join(str(x) for x in report))
print("\nCharts saved to", CHARTS)
