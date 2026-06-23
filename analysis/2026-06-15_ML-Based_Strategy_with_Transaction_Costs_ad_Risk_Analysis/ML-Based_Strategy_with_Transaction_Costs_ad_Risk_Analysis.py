# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL     = 'PG'
START      = '2010-01-01'
END        = '2024-12-31'
LAGS       = 5       # Use the previous 5 days' direction as features
TRAIN_FRAC = 0.8     # First 80% of data used for training
LEVERAGE   = 3       # Leverage ratio (for risk analysis)
EQUITY     = 10000   # Initial equity in USD

# ── Fetch data ────────────────────────────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['returns']   = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)

# ── Build binary lag features ─────────────────────────────────────────────────
# Feature value: 0 = down day, 1 = up day
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

# ── Estimate proportional transaction cost ────────────────────────────────────
# For US equities the typical bid-ask spread is roughly 0.01%–0.05% of price.
# Here we use 0.02% (i.e., cost per one-way trip).
ptc = 0.0002  # Proportional Transaction Cost

# ════════════════════════════════════════════════════════════════════════════════
# SVM strategy: train, predict, and backtest with transaction costs
# ════════════════════════════════════════════════════════════════════════════════
split = int(len(data) * TRAIN_FRAC)
train = data.iloc[:split].copy()
test  = data.iloc[split:].copy()

# Train SVM classifier
model = SVC(C=1, kernel='linear', gamma='auto')
model.fit(train[cols], train['direction'])

# Predict position on the test set
test['position'] = model.predict(test[cols])
print(test['position'].value_counts())

# Accuracy (hit ratio)
train_acc = accuracy_score(train['direction'], model.predict(train[cols]))
test_acc  = accuracy_score(test['direction'], test['position'])
print(f'=== SVM Strategy Accuracy ===')
print(f'Train set: {train_acc:.2%}')
print(f'Test set:  {test_acc:.2%}')

# Strategy return without transaction costs
test['strategy'] = test['position'] * test['returns']

# Number of position changes (roundtrips)
n_trades = (test['position'].diff() != 0).sum()
print(f'\nNumber of trades: {n_trades}')

# Strategy return with transaction costs: deduct cost on every position change
test['strategy_tc'] = np.where(
    test['position'].diff() != 0,        # Days when position changes
    test['strategy'] - ptc,              # Deduct transaction cost
    test['strategy']                     # No change, no cost
)

# ── Total performance ──────────────────────────────────────────────────────────
perf = test[['returns', 'strategy', 'strategy_tc']].sum().apply(np.exp)
print(f'\n=== Test Set Total Performance ===')
print(perf.round(4))

# ── Plot 1: Cumulative performance comparison ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
test[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG SVM Strategy Cumulative Performance (with/without Transaction Costs)')
ax.legend(['Buy & Hold PG (benchmark)', 'SVM Strategy (no cost)', 'SVM Strategy (with cost)'])
plt.tight_layout()
plt.savefig('pg_svm_performance.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# Optimal leverage calculation (Kelly Criterion applied to strategy returns)
# ════════════════════════════════════════════════════════════════════════════════
# Annualized statistics (252 trading days)
mean_ret  = test[['returns', 'strategy_tc']].mean() * 252
var_ret   = test[['returns', 'strategy_tc']].var() * 252

kelly_f   = mean_ret / var_ret
print(f'\n=== Optimal Kelly Leverage ===')
print(kelly_f.round(4))
print(f'\nHalf Kelly:')
print((kelly_f * 0.5).round(4))

# ── Performance simulation under different leverage ratios ─────────────────────
to_plot = ['returns', 'strategy_tc']
for lev in [1, 2, 3, 5]:
    col = f'strategy_tc_lev{lev}'
    test[col] = test['strategy_tc'] * lev
    to_plot.append(col)

fig, ax = plt.subplots(figsize=(12, 5))
test[to_plot].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG SVM Strategy Cumulative Performance under Different Leverage Ratios')
ax.legend(['Buy & Hold PG', 'Strategy (with cost, leverage 1)',
           'Leverage 1x', 'Leverage 2x', 'Leverage 3x', 'Leverage 5x'])
plt.tight_layout()
plt.savefig('pg_svm_leverage.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# Risk analysis
# ════════════════════════════════════════════════════════════════════════════════
# Simulate equity curve with the specified LEVERAGE
risk = pd.DataFrame()
risk['strategy_lev'] = test['strategy_tc'] * LEVERAGE

# Equity curve: starting from EQUITY dollars, compounded by strategy returns
risk['equity'] = risk['strategy_lev'].cumsum().apply(np.exp) * EQUITY

# ── Maximum drawdown ──────────────────────────────────────────────────────────
# Max drawdown = largest drop from a historical peak to the current trough
risk['cummax']   = risk['equity'].cummax()          # Running maximum equity
risk['drawdown'] = risk['cummax'] - risk['equity']  # Drawdown in dollar terms

max_dd   = risk['drawdown'].max()
t_max_dd = risk['drawdown'].idxmax()

print(f'\n=== Risk Analysis (leverage {LEVERAGE}x, initial equity ${EQUITY}) ===')
print(f'Maximum drawdown: ${max_dd:.2f}')
print(f'Date of maximum drawdown: {t_max_dd.date()}')

# Compute the longest drawdown period (time between consecutive equity highs)
highs   = risk['drawdown'][risk['drawdown'] == 0]
print(highs)
if len(highs) > 1:
    periods = (highs.index[1:] - highs.index[:-1]).days
    max_period = periods.max()
    print(f'Longest drawdown period: {max_period} days')

# ── Plot 2: Equity curve and maximum drawdown ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
risk[['equity', 'cummax']].plot(ax=ax, label=['Equity Curve', 'Historical Peak'])
ax.axvline(t_max_dd, color='r', alpha=0.5, linestyle='--', label='Max Drawdown Point')
ax.set_title(f'PG SVM Strategy Equity Curve (Leverage {LEVERAGE}x)')
ax.legend()
plt.tight_layout()
plt.savefig('pg_svm_drawdown.png', dpi=150)
plt.show()

# ── Value at Risk (VaR) ───────────────────────────────────────────────────────
# VaR: the maximum expected loss at a given confidence level over a time horizon
risk['equity_returns'] = np.log(risk['equity'] / risk['equity'].shift(1))
risk.dropna(inplace=True)

percs = np.array([0.01, 0.1, 1.0, 2.5, 5.0, 10.0])  # Percentile levels
VaR   = scs.scoreatpercentile(EQUITY * risk['equity_returns'], percs)

print(f'\n=== VaR (Daily, Confidence Level vs Maximum Loss) ===')
print(f'{"Confidence":>12s} {"Max Loss ($)":>15s}')
print('-' * 30)
for conf, var in zip(100 - percs, -VaR):
    print(f'{conf:>12.2f}% {var:>15.2f}')