# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'
LAGS   = 2  # Use the previous 2 days' log returns as features

# ── Data preparation (same as the OLS regression approach) ───────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['returns']   = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)

cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

# ════════════════════════════════════════════════════════════════════════════════
# Part 1: K-Means Clustering Strategy
# Concept: let the algorithm automatically find two clusters in feature space,
# then determine which cluster corresponds to up moves and which to down moves.
# ════════════════════════════════════════════════════════════════════════════════

# Use lag_1 and lag_2 as features, split into 2 clusters
model_kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
model_kmeans.fit(data[cols])

# Predict which cluster each trading day belongs to (0 or 1)
data['pos_clus'] = model_kmeans.predict(data[cols])

# Map cluster label to position: cluster 1 -> short (-1), cluster 0 -> long (+1)
# This mapping is arbitrary; which cluster represents up moves depends on the data.
data['pos_clus'] = np.where(data['pos_clus'] == 1, -1, 1)

# ── Plot 1: K-Means cluster scatter plot ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1],
           c=data['pos_clus'], cmap='coolwarm', alpha=0.5, s=5)
ax.set_title('PG K-Means Clustering Results (lag_1 vs lag_2)')
ax.set_xlabel("lag_1 (yesterday's return)")
ax.set_ylabel("lag_2 (return two days ago)")
plt.tight_layout()
plt.savefig('pg_kmeans_clusters.png', dpi=150)
plt.show()

# ── Compute K-Means strategy returns ──────────────────────────────────────────
data['strat_clus'] = data['pos_clus'] * data['returns']

perf_clus = data[['returns', 'strat_clus']].sum().apply(np.exp)
hit_clus  = (data['direction'] == data['pos_clus']).mean()
print('=== K-Means Strategy Performance ===')
print(perf_clus.round(4))
print(f'Prediction accuracy: {hit_clus:.2%}')

# ── Plot 2: K-Means strategy cumulative performance ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot(
    ax=ax, label=['Buy & Hold PG (benchmark)', 'K-Means Strategy'])
ax.set_title('PG K-Means Clustering Strategy Cumulative Performance')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kmeans_performance.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# Part 2: Frequency Approach Strategy
# Concept: convert continuous log returns to binary (0/1) and tabulate the
# historical frequency of up moves for each feature combination.
# Use the frequency to decide position: if the combination mostly precedes a
# down move, go short; otherwise, go long.
# ════════════════════════════════════════════════════════════════════════════════

def create_bins(data, cols, bins=[0]):
    """
    Discretize continuous feature values into binary labels.
    bins=[0] means split at 0: negative -> 0 (down), positive -> 1 (up).
    """
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        # np.digitize: maps each value to the interval index defined by bins
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)
    return cols_bin

cols_bin = create_bins(data, cols, bins=[0])

# Tabulate the count of up/down moves for each feature combination
grouped = data.groupby(cols_bin + ['direction'])
freq_table = grouped['direction'].size().unstack(fill_value=0)
print('\n=== Frequency Table (up/down counts per feature combination) ===')
print(freq_table)

# Rule: if both features equal 1 (both previous days went up), predict down (short).
# For all other combinations, predict up (long).
# This rule is derived from observing which combination has a higher historical
# frequency of down moves in the frequency table.
data['pos_freq'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1)

# ── Compute frequency approach strategy returns ───────────────────────────────
data['strat_freq'] = data['pos_freq'] * data['returns']

perf_freq = data[['returns', 'strat_freq']].sum().apply(np.exp)
hit_freq  = (data['direction'] == data['pos_freq']).mean()
print('\n=== Frequency Approach Strategy Performance ===')
print(perf_freq.round(4))
print(f'Prediction accuracy: {hit_freq:.2%}')

# ── Plot 3: Frequency approach strategy cumulative performance ────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_freq']].cumsum().apply(np.exp).plot(
    ax=ax, label=['Buy & Hold PG (benchmark)', 'Frequency Approach Strategy'])
ax.set_title('PG Frequency Approach Strategy Cumulative Performance')
ax.legend()
plt.tight_layout()
plt.savefig('pg_freq_performance.png', dpi=150)
plt.show()

# ── Plot 4: Combined comparison of both strategies ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_clus', 'strat_freq']].cumsum().apply(np.exp).plot(
    ax=ax, label=['Buy & Hold PG (benchmark)', 'K-Means Strategy', 'Frequency Approach Strategy'])
ax.set_title('PG K-Means vs Frequency Approach Strategy Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('pg_cluster_freq_compare.png', dpi=150)
plt.show()