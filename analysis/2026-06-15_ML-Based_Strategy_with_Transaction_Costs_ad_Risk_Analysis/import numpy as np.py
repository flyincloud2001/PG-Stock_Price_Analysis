import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SYMBOL = 'PG'
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
LAG = 5

raw = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = ['SYMBOL']
data['return'] = np.log(data['SYMBOL'] / data['SYMBOL'].shift(1))
data.dropna(inplace=True)
data['direction'] = np.where(data['return'] > 0, 1, 0)

for lag in range(1, LAG + 1):
    data[f'lag_{lag}'] = data['return'].shift(lag)
data.dropna(inplace=True)

