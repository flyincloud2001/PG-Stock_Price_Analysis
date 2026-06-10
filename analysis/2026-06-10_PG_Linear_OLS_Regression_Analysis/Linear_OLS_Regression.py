# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── 參數設定 ──────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'
LAGS   = 2  # 使用前2天的 log return 作為特徵

# ── 抓取資料並計算 log return ─────────────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]

# log return：ln(今日收盤 / 昨日收盤)，用 log return 而非價格是因為 log return 具有定態性
data['returns'] = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)

# direction：只看漲跌方向，+1 表示上漲，-1 表示下跌，0 表示不變
data['direction'] = np.sign(data['returns']).astype(int)

# ── 建立滯後特徵欄位 ──────────────────────────────────────────────────────────
# 概念：若昨天和前天的漲跌能預測今天，就存在可利用的價格規律
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)  # 將 log return 向後平移
    cols.append(col)

data.dropna(inplace=True)

# ── 圖1：log return 分佈直方圖 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
data['returns'].hist(bins=50, ax=ax, color='steelblue', edgecolor='white')
ax.set_title('PG 日 Log Return 分佈')
ax.set_xlabel('Log Return')
ax.set_ylabel('頻率')
plt.tight_layout()
plt.savefig('pg_ols_return_hist.png', dpi=150)
plt.show()

# ── 圖2：lag_1 vs lag_2 散點圖（顏色代表當日報酬）─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(data['lag_1'], data['lag_2'], c=data['returns'],
                cmap='coolwarm', alpha=0.5, s=5)
plt.colorbar(sc, ax=ax, label='當日 Log Return')
ax.axvline(0, color='r', linestyle='--', linewidth=0.8)
ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
ax.set_title('PG lag_1 vs lag_2 散點圖')
ax.set_xlabel('lag_1（昨日報酬）')
ax.set_ylabel('lag_2（前天報酬）')
plt.tight_layout()
plt.savefig('pg_ols_scatter.png', dpi=150)
plt.show()

# ── OLS 回歸策略 ──────────────────────────────────────────────────────────────
model = LinearRegression()

# 方法一：用 log return 作為 label 來訓練回歸
# 預測值是連續數字，再用正負號轉為 +1/-1 倉位
data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols])

# 方法二：用 direction（+1/-1）作為 label 來訓練回歸
# 直接預測漲跌方向，理論上更直接
data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols])

# 將連續預測值轉為倉位訊號（正數→做多+1，負數→做空-1）
data[['pos_ols_1', 'pos_ols_2']] = np.where(
    data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)

# ── 計算策略報酬（vectorized backtesting）────────────────────────────────────
# 策略報酬 = 倉位訊號 × 當日 log return
# 注意：這裡沒有 shift(1)，因為特徵本身已是滯後的（lag_1、lag_2）
# 所以訊號是用「過去的資料」產生的，不存在前視偏差
data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
data['strat_ols_2'] = data['pos_ols_2'] * data['returns']

# ── 印出總績效 ────────────────────────────────────────────────────────────────
perf = data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)
print('=== 總績效（投入1元最終變成幾元）===')
print(perf.round(4))

# 預測準確率（hit ratio）
print('\n=== 預測準確率 ===')
print(f'方法一 (label=returns)：'
      f'{(data["direction"] == data["pos_ols_1"]).mean():.2%}')
print(f'方法二 (label=direction)：'
      f'{(data["direction"] == data["pos_ols_2"]).mean():.2%}')

# ── 圖3：累積績效對比 ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(
    ax=ax,
)
ax.set_title('PG OLS 回歸策略累積績效')
ax.legend(['持有 PG（benchmark）', 'OLS 策略一（label=returns）',
           'OLS 策略二（label=direction）'])
plt.tight_layout()
plt.savefig('pg_ols_performance.png', dpi=150)
plt.show()
