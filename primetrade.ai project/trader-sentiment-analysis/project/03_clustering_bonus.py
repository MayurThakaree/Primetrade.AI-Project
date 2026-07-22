"""
Bonus: cluster traders into behavioral archetypes using KMeans,
based on trade size, frequency, win rate, PnL volatility and long/short bias.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA = '/home/claude/project/data'
CHARTS = '/home/claude/project/charts'
OUT = '/home/claude/project/output'

acct_day = pd.read_csv(f'{DATA}/account_day_metrics.csv')
acct_summary = pd.read_csv(f'{DATA}/account_summary_segments.csv')

ls = acct_day.groupby('Account')['long_short_ratio'].mean().reset_index().rename(
    columns={'long_short_ratio': 'avg_long_short_ratio'})
feat = acct_summary.merge(ls, on='Account', how='left')

feature_cols = ['avg_trade_size', 'avg_daily_trades', 'avg_win_rate', 'pnl_std', 'avg_long_short_ratio']
feat_clean = feat.dropna(subset=feature_cols).copy()

X = StandardScaler().fit_transform(feat_clean[feature_cols])
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
feat_clean['cluster'] = km.fit_predict(X)

profile = feat_clean.groupby('cluster')[feature_cols + ['total_pnl']].mean().round(2)
profile['n_accounts'] = feat_clean.groupby('cluster').size()

# Name clusters by RANKING clusters against each other on each feature (not a fixed
# threshold), so each cluster gets a distinct, relative descriptor.
size_rank = profile['avg_trade_size'].rank(ascending=False)   # 1 = biggest size
freq_rank = profile['avg_daily_trades'].rank(ascending=False)  # 1 = most frequent
wr_rank = profile['avg_win_rate'].rank(ascending=False)        # 1 = highest win rate
vol_rank = profile['pnl_std'].rank(ascending=False)            # 1 = most volatile

labels = {}
for c in profile.index:
    tags = []
    tags.append('large-size' if size_rank[c] <= k / 2 else 'small-size')
    tags.append('high-freq' if freq_rank[c] <= k / 2 else 'low-freq')
    tags.append('high win-rate' if wr_rank[c] <= k / 2 else 'lower win-rate')
    tags.append('volatile PnL' if vol_rank[c] <= k / 2 else 'steady PnL')
    labels[c] = ' / '.join(tags)

feat_clean['archetype'] = feat_clean['cluster'].map(labels)
feat_clean.to_csv(f'{DATA}/trader_clusters.csv', index=False)

report = []
report.append("=== Cluster profiles (mean feature values per cluster) ===")
report.append(profile.to_string())
report.append("\n=== Archetype labels ===")
for c, l in labels.items():
    report.append(f"Cluster {c}: {l}  (n={int(profile.loc[c, 'n_accounts'])})")

with open(f'{OUT}/03_clustering_report.txt', 'w') as f:
    f.write('\n'.join(report))
print('\n'.join(report))

# scatter: trade size vs frequency, colored by cluster
fig, ax = plt.subplots(figsize=(7, 5.5))
cmap = plt.get_cmap('tab10')
for c in sorted(feat_clean['cluster'].unique()):
    sub = feat_clean[feat_clean['cluster'] == c]
    ax.scatter(sub['avg_trade_size'], sub['avg_daily_trades'], s=90,
               color=cmap(c), label=f"Cluster {c}: {labels[c]}", edgecolor='white')
ax.set_xlabel('Avg trade size (USD)')
ax.set_ylabel('Avg trades per day')
ax.set_xscale('log')
ax.set_title('Trader archetypes (KMeans, k=4)')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(f'{CHARTS}/07_trader_archetypes_scatter.png')
plt.close()
print("\nChart saved: 07_trader_archetypes_scatter.png")
