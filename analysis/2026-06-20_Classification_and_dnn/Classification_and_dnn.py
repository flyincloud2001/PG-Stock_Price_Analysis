# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'
LAGS   = 5  # Use the previous 5 days' log returns as features

# ── Data preparation ──────────────────────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['returns']   = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)

# Build 5 lag feature columns
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

# ── Feature discretization helper ─────────────────────────────────────────────
def create_bins(data, cols, bins=[0]):
    """Convert continuous features to discrete values; bins=[0] splits at 0 into 0 and 1."""
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)
    return cols_bin

# ── Compute bins using mean and std (split into 4 intervals) ─────────────────
mu   = data['returns'].mean()
std  = data['returns'].std()
bins = [mu - std, mu, mu + std]  # Split returns into 4 intervals

cols_bin = create_bins(data, cols, bins=bins)

# ════════════════════════════════════════════════════════════════════════════════
# Define all classification models
# ════════════════════════════════════════════════════════════════════════════════
C = 1  # Regularization parameter (smaller = stronger regularization)

models = {
    'log_reg':  linear_model.LogisticRegression(C=C, max_iter=1000),
    'gauss_nb': GaussianNB(),
    'svm':      SVC(C=C, kernel='linear', gamma='auto')
}

def fit_models(train_data):
    """Fit all models on training data."""
    for name, model in models.items():
        model.fit(train_data[cols_bin], train_data['direction'])

def derive_positions(test_data):
    """Generate position predictions for test data using fitted models."""
    for name, model in models.items():
        test_data[f'pos_{name}'] = model.predict(test_data[cols_bin])

def evaluate_strats(test_data):
    """Compute strategy returns for each model."""
    sel = ['returns']
    for name in models.keys():
        col = f'strat_{name}'
        test_data[col] = test_data[f'pos_{name}'] * test_data['returns']
        sel.append(col)
    return sel

# ════════════════════════════════════════════════════════════════════════════════
# Part 1: Sequential Train-Test Split (time-ordered)
# Concept: first 70% for training, last 30% for testing, simulating real trading
# ════════════════════════════════════════════════════════════════════════════════
split      = int(len(data) * 0.7)
train_seq  = data.iloc[:split].copy()
test_seq   = data.iloc[split:].copy()

fit_models(train_seq)
derive_positions(test_seq)
print(test_seq['pos_log_reg'])
print(test_seq['pos_gauss_nb'])
print(test_seq['pos_svm'])
sel_seq = evaluate_strats(test_seq)

print('=== Sequential Split Test Set Performance ===')
print(test_seq[sel_seq].sum().apply(np.exp).round(4))

print(data['direction'].value_counts())
print(test_seq['pos_svm'].value_counts())

print('\n=== Sequential Split Prediction Accuracy ===')
for name in models.keys():
    acc = (test_seq['direction'] == test_seq[f'pos_{name}']).mean()
    print(f'  {name}: {acc:.2%}')

fig, ax = plt.subplots(figsize=(12, 5))
test_seq[sel_seq].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG Classification Strategy Cumulative Performance (Sequential Split)')
ax.legend(['Buy & Hold PG'] + list(models.keys()))
plt.tight_layout()
plt.savefig('pg_cls_sequential.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# Part 2: Randomized Train-Test Split
# Concept: assume price patterns are time-invariant; randomly sample train/test data.
# Note: this assumption does not necessarily hold for financial data.
# ════════════════════════════════════════════════════════════════════════════════
train_rand, test_rand = train_test_split(
    data, test_size=0.5, shuffle=True, random_state=100)

train_rand = train_rand.copy().sort_index()
test_rand  = test_rand.copy().sort_index()

fit_models(train_rand)
derive_positions(test_rand)
sel_rand = evaluate_strats(test_rand)

print('\n=== Randomized Split Test Set Performance ===')
print(test_rand[sel_rand].sum().apply(np.exp).round(4))

fig, ax = plt.subplots(figsize=(12, 5))
test_rand[sel_rand].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG Classification Strategy Cumulative Performance (Randomized Split)')
ax.legend(['Buy & Hold PG'] + list(models.keys()))
plt.tight_layout()
plt.savefig('pg_cls_randomized.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# Part 3: Deep Neural Network (DNN)
# Concept: MLPClassifier is a multi-layer perceptron; hidden_layer_sizes defines
# the architecture. E.g., [256, 256] means two hidden layers with 256 neurons each.
# ════════════════════════════════════════════════════════════════════════════════
dnn_model = MLPClassifier(
    solver='adam',              # Optimizer; adam works well for large datasets
    alpha=1e-4,                 # L2 regularization strength to prevent overfitting
    hidden_layer_sizes=[256, 256],  # Two hidden layers with 256 neurons each
    max_iter=500,               # Maximum training iterations
    random_state=1
)

# Train DNN on the sequential split training set
dnn_model.fit(train_seq[cols_bin], train_seq['direction'])

# Predict on test set
test_seq['pos_dnn'] = dnn_model.predict(test_seq[cols_bin])
test_seq['strat_dnn'] = test_seq['pos_dnn'] * test_seq['returns']

dnn_acc  = (test_seq['direction'] == test_seq['pos_dnn']).mean()
dnn_perf = test_seq[['returns', 'strat_dnn']].sum().apply(np.exp)

print('\n=== DNN Strategy Performance (Sequential Split) ===')
print(dnn_perf.round(4))
print(f'DNN prediction accuracy: {dnn_acc:.2%}')

fig, ax = plt.subplots(figsize=(12, 5))
test_seq[['returns', 'strat_dnn']].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG DNN Strategy Cumulative Performance (Sequential Split)')
ax.legend(['Buy & Hold PG (benchmark)', 'DNN Strategy'])
plt.tight_layout()
plt.savefig('pg_dnn_performance.png', dpi=150)
plt.show()