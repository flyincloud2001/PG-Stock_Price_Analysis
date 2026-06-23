# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL = 'PG'          # Stock ticker for Procter & Gamble
START  = '2010-01-01'  # Data start date
END    = '2024-12-31'  # Data end date
LAGS   = 5             # Number of lags (use the past 5 days to predict today)

# ── Fetch data ────────────────────────────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data.dropna(inplace=True)

# ── Build lag columns ──────────────────────────────────────────────────────────
# Concept: if RWH holds, the coefficient of lag_1 (yesterday's price) should be
# close to 1 and all other lag coefficients close to 0 — meaning today's best
# predictor is simply yesterday's price, so technical analysis has no edge.
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data[SYMBOL].shift(lag)  # Shift price backward by lag days
    cols.append(col)

data.dropna(inplace=True)

# ── OLS regression: predict today's price from the past 5 days ───────────────
# np.linalg.lstsq solves the least-squares problem to find the best coefficients
# Formula: price_today ≈ beta_0 + beta_1*lag1 + beta_2*lag2 + ... + beta_5*lag5
X = np.column_stack([np.ones(len(data)), data[cols].values])  # Add intercept term
y = data[SYMBOL].values

# lstsq returns (coefficients, residuals, rank, singular values); we only need coefficients
beta = np.linalg.lstsq(X, y, rcond=None)[0]

# ── Plot 1: Regression coefficient bar chart ──────────────────────────────────
# If RWH holds, lag_1 coefficient ≈ 1 and the rest ≈ 0
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(['intercept'] + cols, beta, color='steelblue')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('PG OLS Regression Coefficients (Predicting Today from Past 5 Days)')
ax.set_xlabel('Lag Term')
ax.set_ylabel('Coefficient Value')
plt.tight_layout()
plt.savefig('pg_rwh_coefficients.png', dpi=150)
plt.show()

# ── Compute predictions ───────────────────────────────────────────────────────
# Apply regression coefficients to get the predicted price for each day
data['prediction'] = X @ beta  # Matrix multiplication: X times beta vector

# ── Plot 2: Actual price vs predicted price ───────────────────────────────────
# If the prediction line nearly equals the actual line shifted one day to the
# right, lag_1 dominates, supporting the RWH.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data[SYMBOL],       label='PG Actual Price',    linewidth=1.2)
ax.plot(data.index, data['prediction'], label='OLS Predicted Price', linewidth=1.0,
        linestyle='--', alpha=0.8)
ax.set_title('PG Actual Price vs OLS Predicted Price')
ax.legend()
plt.tight_layout()
plt.savefig('pg_rwh_prediction.png', dpi=150)
plt.show()

# ── Print regression coefficients ────────────────────────────────────────────
print('=== OLS Regression Coefficients ===')
labels = ['intercept'] + [f'lag_{i}' for i in range(1, LAGS + 1)]
for label, coef in zip(labels, beta):
    print(f'  {label:>10s}: {coef:.6f}')

# ── Interpretation ────────────────────────────────────────────────────────────
# If lag_1 coefficient ≈ 1 and the rest ≈ 0:
#   → Supports RWH: yesterday's price is the best predictor; past info adds nothing
#   → Technical indicators (SMA, RSI, etc.) should not generate excess returns in theory
# If other lag coefficients are noticeably non-zero:
#   → Weak-form EMH may not hold completely; exploitable price patterns may exist
print('\n=== Interpretation ===')
print(f'lag_1 coefficient: {beta[1]:.6f} (close to 1 supports RWH)')
print(f'Average absolute value of lag_2 to lag_5 coefficients: '
      f'{np.mean(np.abs(beta[2:])):.6f} (close to 0 supports RWH)')