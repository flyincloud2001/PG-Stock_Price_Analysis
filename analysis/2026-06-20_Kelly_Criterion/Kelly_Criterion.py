# ── Imports ───────────────────────────────────────────────────────────────────
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'

# ════════════════════════════════════════════════════════════════════════════════
# Part 1: Kelly Criterion in a binary gambling simulation
# Concept: in a game with win probability p > 0.5, what fraction of capital
# should be wagered each round to maximize long-run wealth?
# Formula: f* = p - q = p - (1 - p) = 2p - 1
# ════════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

p = 0.55        # Win probability 55%
f_star = p - (1 - p)  # Optimal Kelly fraction = 0.10
I = 50          # Simulate 50 paths
n = 200         # Each path has 200 trials

def run_simulation(f, p=0.55, n=200, I=50):
    """
    Simulate a binary gambling game.
    f: fraction of capital wagered per round
    Returns an (n, I) matrix where each column is one simulated path.
    """
    c = np.zeros((n, I))
    c[0] = 100  # Initial capital: $100
    for i in range(I):
        for t in range(1, n):
            outcome = np.random.binomial(1, p)  # 1 = win, 0 = loss
            if outcome > 0:
                c[t, i] = (1 + f) * c[t - 1, i]  # Win: capital increases
            else:
                c[t, i] = (1 - f) * c[t - 1, i]  # Loss: capital decreases
    return c

# Simulate under different f values
c_optimal   = run_simulation(f_star)        # f* = 0.10
c_half      = run_simulation(f_star * 0.5)  # Half Kelly = 0.05
c_double    = run_simulation(f_star * 2.5)  # Over-betting = 0.25
c_extreme   = run_simulation(0.5)           # Extreme over-betting

# ── Plot 1: All simulation paths under optimal Kelly ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(c_optimal, 'b', lw=0.3, alpha=0.5)
ax.plot(c_optimal.mean(axis=1), 'r', lw=2, label='Average equity curve')
ax.set_title(f'Binary Gambling Simulation (f* = {f_star:.2f}, win prob = {p})')
ax.set_xlabel('Trial Number')
ax.set_ylabel('Capital')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kelly_simulation.png', dpi=150)
plt.show()

# ── Plot 2: Average equity curves under different f values ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(c_optimal.mean(axis=1), label=f'f* = {f_star:.2f} (optimal)')
ax.plot(c_half.mean(axis=1),    label=f'f = {f_star*0.5:.2f} (half Kelly)')
ax.plot(c_double.mean(axis=1),  label=f'f = {f_star*2.5:.2f} (over-betting)')
ax.plot(c_extreme.mean(axis=1), label='f = 0.50 (extreme)')
ax.set_title('Average Equity Curves under Different f Values')
ax.set_xlabel('Trial Number')
ax.set_ylabel('Average Capital')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kelly_comparison.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# Part 2: Kelly Criterion applied to PG stock
# Formula: f* = (mu - r) / sigma^2
# mu: annualized expected return, r: risk-free rate, sigma^2: annualized variance
# ════════════════════════════════════════════════════════════════════════════════
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['returns'] = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)

# Compute annualized statistics (252 trading days)
mu    = data['returns'].mean() * 252       # Annualized mean return
sigma = data['returns'].std() * np.sqrt(252)  # Annualized volatility
r     = 0.0                                # Simplified: risk-free rate = 0

# Optimal Kelly leverage fraction
f_pg = (mu - r) / sigma ** 2

print(f'=== PG Kelly Criterion ===')
print(f'Annualized mean return mu:  {mu:.4f}')
print(f'Annualized volatility sigma: {sigma:.4f}')
print(f'Optimal Kelly leverage f*:  {f_pg:.4f}')
print(f'Half Kelly leverage:        {f_pg * 0.5:.4f}')

# ── Simulate Kelly leverage strategy equity curve ─────────────────────────────
def kelly_strategy(data, symbol, f):
    """
    Simulate a Kelly leveraged strategy.
    Each day the capital position is rebalanced based on today's return.
    equ: equity (net worth)
    cap: deployed capital (= equ * f)
    """
    equ_col = f'equity_{f:.2f}'
    cap_col = f'capital_{f:.2f}'
    data[equ_col] = 1.0   # Initial equity set to 1
    data[cap_col] = f      # Initial capital deployed = 1 * f

    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]
        # New capital = old capital * exp(today's return)
        data.loc[t, cap_col] = (data.loc[t_1, cap_col] *
                                 math.exp(data.loc[t, 'returns']))
        # New equity = old equity + capital gain/loss
        data.loc[t, equ_col] = (data.loc[t_1, equ_col] +
                                  data.loc[t, cap_col] -
                                  data.loc[t_1, cap_col])
        # Rebalance deployed capital to f * new equity
        data.loc[t, cap_col] = data.loc[t, equ_col] * f

    return equ_col

eq_full = kelly_strategy(data, SYMBOL, f_pg)        # Full Kelly
eq_half = kelly_strategy(data, SYMBOL, f_pg * 0.5)  # Half Kelly
eq_qtr  = kelly_strategy(data, SYMBOL, f_pg * 0.25) # Quarter Kelly

# ── Plot 3: PG Kelly strategy vs buy-and-hold ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data['returns'].cumsum().apply(np.exp).plot(ax=ax, label='Buy & Hold PG')
data[[eq_full, eq_half, eq_qtr]].plot(ax=ax,
    label=[f'Full Kelly (f={f_pg:.2f})',
           f'Half Kelly (f={f_pg*0.5:.2f})',
           f'Quarter Kelly (f={f_pg*0.25:.2f})'])
ax.set_title(f'PG Kelly Leverage Strategy vs Buy & Hold')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kelly_equity.png', dpi=150)
plt.show()