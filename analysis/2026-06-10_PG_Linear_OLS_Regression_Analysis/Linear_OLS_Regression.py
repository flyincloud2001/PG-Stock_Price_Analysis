# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'
LAGS   = 2  # Use the previous 2 days' log returns as features

# ── Fetch data and compute log returns ────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]

# Log return: ln(today's close / yesterday's close)
# Log returns are used instead of prices because they are stationary
data['returns'] = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)

# Direction: sign of log return (+1 = up, -1 = down, 0 = flat)
data['direction'] = np.sign(data['returns']).astype(int)

# ── Build lag feature columns ─────────────────────────────────────────────────
# Concept: if yesterday's and the day before's moves can predict today's,
# there is an exploitable price pattern.
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)  # Shift log return backward
    cols.append(col)

data.dropna(inplace=True)

# ── Plot 1: Log return distribution histogram ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
data['returns'].hist(bins=50, ax=ax, color='steelblue', edgecolor='white')
ax.set_title('PG Daily Log Return Distribution')
ax.set_xlabel('Log Return')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('pg_ols_return_hist.png', dpi=150)
plt.show()

# ── Plot 2: Scatter plot of lag_1 vs lag_2 (colored by today's return) ───────
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(data['lag_1'], data['lag_2'], c=data['returns'],
                cmap='coolwarm', alpha=0.5, s=5)
plt.colorbar(sc, ax=ax, label="Today's Log Return")
ax.axvline(0, color='r', linestyle='--', linewidth=0.8)
ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
ax.set_title('PG lag_1 vs lag_2 Scatter Plot')
ax.set_xlabel("lag_1 (yesterday's return)")
ax.set_ylabel("lag_2 (return two days ago)")
plt.tight_layout()
plt.savefig('pg_ols_scatter.png', dpi=150)
plt.show()

# ── OLS regression strategy ───────────────────────────────────────────────────
model = LinearRegression()

# Method 1: use log return as label for regression
# Predictions are continuous; convert to +1/-1 position via sign
data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols])

# Method 2: use direction (+1/-1) as label for regression
# Directly predicts the up/down direction
data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols])

# Convert continuous predictions to position signals (positive -> long +1, negative -> short -1)
data[['pos_ols_1', 'pos_ols_2']] = np.where(
    data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)

# ── Compute strategy returns (vectorized backtesting) ─────────────────────────
# Strategy return = position signal x today's log return
# No shift(1) needed here because the features are already lagged (lag_1, lag_2),
# so the signal is generated from past data only — no look-ahead bias.
data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
data['strat_ols_2'] = data['pos_ols_2'] * data['returns']

# ── Print total performance ───────────────────────────────────────────────────
perf = data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)
print('=== Total Performance (final value of $1 invested) ===')
print(perf.round(4))

# Prediction accuracy (hit ratio)
print('\n=== Prediction Accuracy ===')
print(f'Method 1 (label=returns):   '
      f'{(data["direction"] == data["pos_ols_1"]).mean():.2%}')
print(f'Method 2 (label=direction): '
      f'{(data["direction"] == data["pos_ols_2"]).mean():.2%}')

# ── Plot 3: Cumulative performance comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(
    ax=ax,
)
ax.set_title('PG OLS Regression Strategy Cumulative Performance')
ax.legend(['Buy & Hold PG (benchmark)', 'OLS Strategy 1 (label=returns)',
           'OLS Strategy 2 (label=direction)'])
plt.tight_layout()
plt.savefig('pg_ols_performance.png', dpi=150)
plt.show()