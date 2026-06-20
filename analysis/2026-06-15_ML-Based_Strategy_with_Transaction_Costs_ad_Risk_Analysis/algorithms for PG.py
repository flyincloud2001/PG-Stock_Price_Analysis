import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


SYMBOL = 'PG'
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
LAG = 5
SPLIT_RATIO = 0.8

raw = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = ['SYMBOL']
data['returns'] = np.log(data['SYMBOL'] / data['SYMBOL'].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)
print(np.sign(data['returns']))        

cols = []
for lag in range(1, LAG + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

def create_bin(data, col, bins=[0]):
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins)
        cols_bin.append(col_bin)
    return cols_bin

cols_bin = create_bin(data, cols)

models = {
    'svm': SVC(C=1, kernel='linear', gamma='auto'), 
    'gauss_nb': GaussianNB(),
    'logistic_regression': LogisticRegression(C=1, max_iter=1000),
    'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
}

def fit_models(train_data):
    for name, model in models.items():
        model.fit(train_data[cols_bin], train_data['direction'])

def derive_position(test_data):
    for name, model in models.items():
        test_data[f'pos_{name}'] = model.predict(test_data[cols_bin])

def evaluate_strate(test_data):
    for name, model in models.items():
        col = f'strat_{name}'
        test_data[col] = test_data[f'pos_{name}'] * test_data['returns']

split = int(len(data)*SPLIT_RATIO)
train_data = data.iloc[:split].copy()
test_data = data.iloc[split:].copy()

fit_models(train_data)
derive_position(test_data)
evaluate_strate(test_data)

fig, ax = plt.subplots(figsize=(12, 6))

