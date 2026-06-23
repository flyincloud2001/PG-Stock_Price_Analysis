# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import product


# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL = 'PG'           # Procter & Gamble stock ticker
START  = '2010-01-01'   # Data start date
END    = '2024-12-31'   # Data end date
SMA1_DEFAULT = 42       # Default short-period SMA window
SMA2_DEFAULT = 252      # Default long-period SMA window
SPLIT_RATIO  = 0.7      # In-sample fraction (70% train, 30% test)

# ── Fetch data ────────────────────────────────────────────────────────────────
raw = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)

# Keep only closing prices and drop missing values
data_full = pd.DataFrame(raw['Close'])
data_full.columns = [SYMBOL]
data_full.dropna(inplace=True)

# ── Split in-sample / out-of-sample ──────────────────────────────────────────
split_idx   = int(len(data_full) * SPLIT_RATIO)
data_in     = data_full.iloc[:split_idx].copy()   # Training period
data_out    = data_full.iloc[split_idx:].copy()   # Testing period

# ── Helper: run SMA strategy and return performance DataFrame ─────────────────
def run_strategy(df: pd.DataFrame, sma1: int, sma2: int) -> pd.DataFrame:
    """
    Given a price DataFrame and two SMA windows,
    return a DataFrame with Returns, Position, and Strategy columns.
    """
    d = df.copy()
    # Log return: ln(today's close / yesterday's close)
    d['Returns']  = np.log(d[SYMBOL] / d[SYMBOL].shift(1))
    # Compute short and long moving averages
    d['SMA1']     = d[SYMBOL].rolling(sma1).mean()
    d['SMA2']     = d[SYMBOL].rolling(sma2).mean()
    d.dropna(inplace=True)
    # Position signal: go long (+1) when short MA > long MA, else short (-1)
    d['Position'] = np.where(d['SMA1'] > d['SMA2'], 1, -1)
    # Strategy return: use previous day's position to avoid look-ahead bias
    d['Strategy'] = d['Position'].shift(1) * d['Returns']
    d.dropna(inplace=True)
    return d

# ── Plot 1: Price + two moving averages ──────────────────────────────────────
data_plot = run_strategy(data_full, SMA1_DEFAULT, SMA2_DEFAULT)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_plot[SYMBOL], label='PG Closing Price', linewidth=1)
ax.plot(data_plot['SMA1'], label=f'SMA{SMA1_DEFAULT}', linewidth=1.2)
ax.plot(data_plot['SMA2'], label=f'SMA{SMA2_DEFAULT}', linewidth=1.2)
ax.set_title('PG Price with Two Simple Moving Averages')
ax.legend()
plt.tight_layout()
plt.savefig('pg_sma_price.png', dpi=150)
plt.show()

# ── Plot 2: Position changes ──────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
ax1.plot(data_plot[SYMBOL], color='steelblue', linewidth=1, label='PG Closing Price')
ax2.plot(data_plot['Position'], color='orange', linewidth=0.8,
        linestyle='--', label='Position')
ax1.set_title('PG Price and Position Signal')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('pg_sma_position.png', dpi=150)
plt.show()

# ── Plot 3: Cumulative performance comparison (in-sample) ─────────────────────
data_in_result = run_strategy(data_in, SMA1_DEFAULT, SMA2_DEFAULT)

# Cumulate log returns then exponentiate to get the wealth multiplier
cumulative = data_in_result[['Returns', 'Strategy']].cumsum().apply(np.exp)

fig, ax = plt.subplots(figsize=(12, 5))
cumulative['Returns'].plot(ax=ax, label='Buy & Hold PG (benchmark)')
cumulative['Strategy'].plot(ax=ax, label='SMA Strategy')
ax.set_title('Cumulative Performance Comparison (In-Sample Training Period)')
ax.legend()
plt.tight_layout()
plt.savefig('pg_sma_performance.png', dpi=150)
plt.show()

# Print total performance
final_perf = np.exp(data_in_result[['Returns', 'Strategy']].sum())
print('=== In-Sample Total Performance (final value of $1 invested) ===')
print(final_perf.round(4))

# ── Brute-force parameter optimization (in-sample only) ──────────────────────
sma1_range = range(20, 61, 4)    # Short SMA: 20 to 60, step 4
sma2_range = range(180, 281, 10) # Long SMA: 180 to 280, step 10

results = []
for s1, s2 in product(sma1_range, sma2_range):
    d = run_strategy(data_in, s1, s2)
    if len(d) == 0:
        continue
    perf = np.exp(d[['Returns', 'Strategy']].sum())
    results.append({
        'SMA1':     s1,
        'SMA2':     s2,
        'MARKET':   round(perf['Returns'], 4),
        'STRATEGY': round(perf['Strategy'], 4),
        'OUT':      round(perf['Strategy'] - perf['Returns'], 4)
    })

results_df = pd.DataFrame(results)

print('\n=== Parameter Optimization Results (Top 7, sorted by excess return) ===')
print(results_df.sort_values('OUT', ascending=False).head(7).to_string(index=False))

# ── Validate best parameters on out-of-sample data ───────────────────────────
best_row = results_df.sort_values('OUT', ascending=False).iloc[0]
best_sma1 = int(best_row['SMA1'])
best_sma2 = int(best_row['SMA2'])

data_out_result = run_strategy(data_out, best_sma1, best_sma2)
cumulative_out  = data_out_result[['Returns', 'Strategy']].cumsum().apply(np.exp)

fig, ax = plt.subplots(figsize=(12, 5))
cumulative_out['Returns'].plot(ax=ax, label='Buy & Hold PG (benchmark)')
cumulative_out['Strategy'].plot(ax=ax, label=f'SMA Strategy ({best_sma1}/{best_sma2})')
ax.set_title(f'Out-of-Sample Validation (SMA1={best_sma1}, SMA2={best_sma2})')
ax.legend()
plt.tight_layout()
plt.savefig('pg_sma_oos.png', dpi=150)
plt.show()

final_oos = np.exp(data_out_result[['Returns', 'Strategy']].sum())
print(f'\n=== Out-of-Sample Performance (SMA1={best_sma1}, SMA2={best_sma2}) ===')
print(final_oos.round(4))