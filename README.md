About Project

```
.
├── data_raw/                        #
│   ├── fear_greed_index.csv
│   └── historical_data.csv
├── data/                            
├── charts/                         
├── output/                          
├── notebooks/
│   ├── trader_sentiment_analysis.ipynb  
│   ├── 01_data_prep.py              
│   ├── 02_analysis.py                
│   └── 03_clustering_bonus.py        
├── WRITEUP.md                        
└── README.md
```

The notebook is the primary deliverable and already contains executed outputs and charts,
so it can be read top-to-bottom without re-running anything. The `.py` scripts under
`notebooks/` contain identical logic and are provided for quick command-line reproduction.

## Setup

```bash
python3 -m pip install pandas numpy matplotlib scikit-learn
```

(Optionally `jupyter` if you want to open/re-run the `.ipynb` interactively:
`pip install jupyter`.)

## How to run

**Option A — open the notebook (recommended):**
```bash
jupyter notebook notebooks/trader_sentiment_analysis.ipynb
```
All outputs/charts are already embedded, but you can re-run all cells (Kernel → Restart &
Run All) — the notebook reads CSVs from `../data_raw/` relative to its own location.

**Option B — run the plain scripts in order:**
```bash
cd notebooks
python3 01_data_prep.py          # Part A: cleans data, builds account-day & day-level metrics -> ../data/
python3 02_analysis.py           # Part B: sentiment vs performance/behavior, segmentation -> ../charts/, ../output/
python3 03_clustering_bonus.py   # Bonus: KMeans trader archetypes -> ../charts/, ../output/, ../data/
```
Note: the scripts read raw CSVs from `/mnt/user-data/uploads/` by default (the environment
they were built in) — edit `RAW_HIST` / `RAW_FG` at the top of `01_data_prep.py` to point at
`../data_raw/historical_data.csv` and `../data_raw/fear_greed_index.csv` if running elsewhere.

## Data notes / limitations

- **No `leverage` column exists** in `historical_data.csv` (no margin/equity field to derive
  it from), despite the assignment description mentioning one. Position notional (**Size
  USD**) is used throughout as the proxy for trade risk/intensity — flagged wherever used.
- Only **32 unique trading accounts** are present across 211k+ trade executions and 246
  coins — segment/cluster conclusions are directional, not statistically robust given the
  small account count.
- Sentiment is a single BTC-wide daily label, applied uniformly across all 246 traded coins.
- "Win rate" / "realized PnL" are computed only over rows with `Closed PnL != 0`
  (position-closing events); pure opens are excluded since they carry no realized PnL yet.

See `WRITEUP.md` for the methodology summary, key insights, and strategy recommendations.
