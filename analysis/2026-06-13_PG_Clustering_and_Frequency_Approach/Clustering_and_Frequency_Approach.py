# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── 參數設定 ──────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'
LAGS   = 2  # 使用前2天的 log return 作為特徵

# ── 資料準備（與概念2相同）───────────────────────────────────────────────────
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
# 第一部分：K-Means 分群策略
# 概念：讓演算法自動在特徵空間中找出兩個群，再判斷哪個群對應上漲、哪個對應下跌
# ════════════════════════════════════════════════════════════════════════════════

# 用 lag_1 和 lag_2 作為特徵，分成2群
model_kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
model_kmeans.fit(data[cols])

# 預測每個交易日屬於哪個群（0 或 1）
data['pos_clus'] = model_kmeans.predict(data[cols])

# 將群號轉為倉位：群 1 → 做空(-1)，群 0 → 做多(+1)
# 這個對應關係是任意的，實際上哪個群代表上漲需要看資料
data['pos_clus'] = np.where(data['pos_clus'] == 1, -1, 1)

# ── 圖1：K-Means 分群散點圖 ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1],
           c=data['pos_clus'], cmap='coolwarm', alpha=0.5, s=5)
ax.set_title('PG K-Means 分群結果（lag_1 vs lag_2）')
ax.set_xlabel('lag_1（昨日報酬）')
ax.set_ylabel('lag_2（前天報酬）')
plt.tight_layout()
plt.savefig('pg_kmeans_clusters.png', dpi=150)
plt.show()

# ── 計算 K-Means 策略報酬 ─────────────────────────────────────────────────────
data['strat_clus'] = data['pos_clus'] * data['returns']

perf_clus = data[['returns', 'strat_clus']].sum().apply(np.exp)
hit_clus  = (data['direction'] == data['pos_clus']).mean()
print('=== K-Means 策略績效 ===')
print(perf_clus.round(4))
print(f'預測準確率：{hit_clus:.2%}')

# ── 圖2：K-Means 策略累積績效 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot(
    ax=ax, label=['持有 PG（benchmark）', 'K-Means 策略'])
ax.set_title('PG K-Means 分群策略累積績效')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kmeans_performance.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# 第二部分：頻率法策略
# 概念：把連續的 log return 轉成二元值（0/1），統計不同組合下上漲的歷史頻率
# 再根據頻率決定倉位：多數情況下跌 → 做空，多數情況上漲 → 做多
# ════════════════════════════════════════════════════════════════════════════════

def create_bins(data, cols, bins=[0]):
    """
    將連續特徵值離散化為二元值。
    bins=[0] 代表以 0 為分界：負數 → 0（下跌），正數 → 1（上漲）
    """
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        # np.digitize：將數值對應到 bins 所定義的區間編號
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)
    return cols_bin

cols_bin = create_bins(data, cols, bins=[0])

# 統計每種特徵組合下的漲跌頻率
grouped = data.groupby(cols_bin + ['direction'])
freq_table = grouped['direction'].size().unstack(fill_value=0)
print('\n=== 頻率表（各特徵組合下的漲跌次數）===')
print(freq_table)

# 規則：若兩個特徵都是 1（前兩天都上漲），則預測下跌（做空）
# 其餘情況預測上漲（做多）
# 這個規則來自頻率表中哪個組合對應下跌機率較高
data['pos_freq'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1)
# ── 計算頻率法策略報酬 ────────────────────────────────────────────────────────
data['strat_freq'] = data['pos_freq'] * data['returns']

perf_freq = data[['returns', 'strat_freq']].sum().apply(np.exp)
hit_freq  = (data['direction'] == data['pos_freq']).mean()
print('\n=== 頻率法策略績效 ===')
print(perf_freq.round(4))
print(f'預測準確率：{hit_freq:.2%}')

# ── 圖3：頻率法策略累積績效 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_freq']].cumsum().apply(np.exp).plot(
    ax=ax, label=['持有 PG（benchmark）', '頻率法策略'])
ax.set_title('PG 頻率法策略累積績效')
ax.legend()
plt.tight_layout()
plt.savefig('pg_freq_performance.png', dpi=150)
plt.show()

# ── 圖4：兩種策略綜合比較 ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data[['returns', 'strat_clus', 'strat_freq']].cumsum().apply(np.exp).plot(
    ax=ax, label=['持有 PG（benchmark）', 'K-Means 策略', '頻率法策略'])
ax.set_title('PG K-Means vs 頻率法策略比較')
ax.legend()
plt.tight_layout()
plt.savefig('pg_cluster_freq_compare.png', dpi=150)
plt.show()
