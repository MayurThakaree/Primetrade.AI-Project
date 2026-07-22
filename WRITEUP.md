# Write-up: Trader Performance vs Market Sentiment

## Methodology

Two datasets were merged on calendar date: `fear_greed_index.csv` (daily BTC sentiment,
2018–2025) and `historical_data.csv` (211,224 individual trade executions across 32
Hyperliquid accounts and 246 coins, 2023–2025). Both were clean on load — zero missing
values, zero duplicate rows. Only 6 trades (0.00%) had no matching sentiment day and were
dropped. The 5-way sentiment classification (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)
was collapsed to a simpler Fear/Neutral/Greed grouping for clearer comparisons.

Two metric tables were built: an **account-day** table (one row per trader per day — trade
count, volume, realized PnL, win rate, long/short ratio, avg trade size) and a **day-level**
table (market-wide daily aggregates). "Realized PnL" and "win rate" only count trades with
`Closed PnL != 0` (i.e., position-closing events), since opening trades carry no PnL yet.
No leverage/margin column exists in the raw data, so **position notional (Size USD)** is
used as a proxy for trade risk/intensity throughout.

Segmentation split the 32 accounts into two-way groups by average trade size (proxy for
risk appetite) and by average daily trade count (frequency), plus a three-way "consistency"
split (consistent winner / inconsistent winner / net loser) based on the ratio of daily PnL
volatility to total profit. A bonus KMeans clustering (k=4, standardized features: trade
size, frequency, win rate, PnL volatility, long/short ratio) was added to find behavioral
archetypes without imposing a fixed segmentation scheme.

## Key insights

1. **Fear days look more profitable on average, but that's driven by tail risk, not skill.**
   Average daily realized PnL per account is higher on Fear days (~$5,185) than Greed days
   (~$4,144), but the 5th-percentile (worst-case) daily PnL is roughly **20x deeper** on Fear
   days (~ -$3,485) than on Greed days (~ -$174). Median PnL — a more representative
   measure, less skewed by outliers — is actually *lower* on Fear days ($123 vs $265). Fear
   days are a fatter-tailed, higher-variance regime, not simply a "better" one.

2. **The Fear-day premium is concentrated in one segment: high-notional traders.** Splitting
   accounts by average trade size, high-size traders earn ~$9,540/day on Fear days vs
   ~$3,347 on Greed days (a ~3x swing) — while low-size traders barely move (~$2,576 on
   Fear vs ~$4,590 on Greed, actually *better* in Greed). The aggregate "Fear days pay more"
   result is not a market-wide effect; it's one trader profile capturing it, and it comes
   with materially larger downside.

3. **Behavior clearly shifts with sentiment: traders size up, trade more, and buy more on
   Fear days.** Avg trade notional is ~$8,530 on Fear days vs ~$5,955 on Greed days; trade
   frequency is ~105 trades/account/day vs ~77; and the long/short (buy/sell) ratio rises
   from ~1.6 to ~2.2. Traders are adding risk into weakness rather than pulling back —
   consistent with the fatter PnL tails observed above.

4. **Most profitable traders are "inconsistent winners," not steady ones.** Of 32 accounts,
   the majority of net-positive traders have high daily-PnL volatility relative to their
   total profit — i.e., a handful of large days drive most of the profit rather than
   broad, repeatable skill. Only a minority qualify as low-volatility "consistent winners."

5. **Clustering surfaces a small set of distinct archetypes**, including a large-size /
   lower-win-rate / high-volatility group (the segment capturing the Fear-day premium above)
   and a small-size / high-win-rate / steady-PnL group that behaves far more conservatively
   regardless of sentiment. One account is a clear outlier (~757 trades/day), likely
   automated execution rather than discretionary trading.

## Strategy recommendations

**1. Segment-aware position sizing on Fear days.** Don't apply a blanket "size up in Fear"
rule. For traders/segments that already run high notional, Fear days are where the edge
shows up — but pair any further size-up with a hard stop-loss, since the downside tail is
~20x deeper than on Greed days. For low-size/low-frequency traders, there's no evidence
Fear days help — avoid chasing size increases in that regime.

**2. Gate frequency increases by win-rate archetype, not sentiment alone.** Frequent traders
show the largest swings between Fear and Greed days, meaning frequency amplifies whatever
sentiment bias is present. Increasing trade frequency during Fear regimes should be reserved
for archetypes with above-median win rates (e.g. the small-size/high-win-rate cluster);
for lower-win-rate/high-volatility clusters, higher frequency during Fear mainly adds
variance without improving outcomes.

## Limitations

- No leverage/margin data — Size USD is a proxy, not true leverage.
- Only 32 accounts — segment and cluster results are directional, not statistically robust.
- Single BTC-wide sentiment label applied uniformly across 246 different coins.
