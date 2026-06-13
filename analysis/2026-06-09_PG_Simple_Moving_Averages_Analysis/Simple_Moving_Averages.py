# 匯入所需套件
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import product


# ── 參數設定 ──────────────────────────────────────────────
SYMBOL = 'PG'           # 寶僑公司股票代碼
START  = '2010-01-01'   # 資料起始日
END    = '2024-12-31'   # 資料結束日
SMA1_DEFAULT = 42       # 短期均線預設天數
SMA2_DEFAULT = 252      # 長期均線預設天數
SPLIT_RATIO  = 0.7      # in-sample 佔比（70% 訓練、30% 測試）

# ── 抓取資料 ──────────────────────────────────────────────
raw = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)

# 只取收盤價，並去除缺失值
data_full = pd.DataFrame(raw['Close'])
data_full.columns = [SYMBOL]
data_full.dropna(inplace=True)

# ── 切分 in-sample / out-of-sample ───────────────────────
split_idx   = int(len(data_full) * SPLIT_RATIO)
data_in     = data_full.iloc[:split_idx].copy()   # 訓練段
data_out    = data_full.iloc[split_idx:].copy()   # 測試段

# ── 函式：計算 SMA 策略並回傳績效 ────────────────────────
def run_strategy(df: pd.DataFrame, sma1: int, sma2: int) -> pd.DataFrame:
    """
    給定價格 DataFrame 與兩個 SMA 天數，
    回傳含有 Returns、Position、Strategy 欄位的 DataFrame。
    """
    d = df.copy()
    # 計算 log return：ln(今日收盤 / 昨日收盤)
    d['Returns']  = np.log(d[SYMBOL] / d[SYMBOL].shift(1))
    # 計算短期與長期均線
    d['SMA1']     = d[SYMBOL].rolling(sma1).mean()
    d['SMA2']     = d[SYMBOL].rolling(sma2).mean()
    d.dropna(inplace=True)
    # 產生倉位訊號：短線 > 長線做多(+1)，否則做空(-1)
    d['Position'] = np.where(d['SMA1'] > d['SMA2'], 1, -1)
    # 策略報酬：用前一日倉位乘以今日報酬，避免前視偏差
    d['Strategy'] = d['Position'].shift(1) * d['Returns']
    d.dropna(inplace=True)
    return d

# ── 圖1：股價 + 兩條均線 ──────────────────────────────────
data_plot = run_strategy(data_full, SMA1_DEFAULT, SMA2_DEFAULT)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_plot[SYMBOL], label='PG 收盤價', linewidth=1)
ax.plot(data_plot['SMA1'], label=f'SMA{SMA1_DEFAULT}', linewidth=1.2)
ax.plot(data_plot['SMA2'], label=f'SMA{SMA2_DEFAULT}', linewidth=1.2)
ax.set_title('PG 股價與兩條簡單移動平均線')
ax.legend()
plt.tight_layout()
plt.savefig('pg_sma_price.png', dpi=150)
plt.show()

# ── 圖2：倉位變化 ─────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
ax1.plot(data_plot[SYMBOL], color='steelblue', linewidth=1, label='PG 收盤價')
ax2.plot(data_plot['Position'], color='orange', linewidth=0.8,
        linestyle='--', label='倉位')
ax1.set_title('PG 股價與倉位訊號')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('pg_sma_position.png', dpi=150)
plt.show()

# ── 圖3：累積績效對比（in-sample）────────────────────────
data_in_result = run_strategy(data_in, SMA1_DEFAULT, SMA2_DEFAULT)

# 將 log return 逐日加總再取 exp，得到累積倍數
cumulative = data_in_result[['Returns', 'Strategy']].cumsum().apply(np.exp)

fig, ax = plt.subplots(figsize=(12, 5))
cumulative['Returns'].plot(ax=ax, label='持有 PG（benchmark）')
cumulative['Strategy'].plot(ax=ax, label='SMA 策略')
ax.set_title('累積績效對比（in-sample 訓練段）')
ax.legend()
plt.tight_layout()
plt.savefig('pg_sma_performance.png', dpi=150)
plt.show()

# 印出總績效數字
final_perf = np.exp(data_in_result[['Returns', 'Strategy']].sum())
print('=== In-Sample 總績效（投入1元最終變成幾元）===')
print(final_perf.round(4))

# ── 參數暴力優化（僅在 in-sample 上跑）──────────────────
sma1_range = range(20, 61, 4)    # 短期均線：20 到 60，每隔 4 天
sma2_range = range(180, 281, 10) # 長期均線：180 到 280，每隔 10 天

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

print('\n=== 參數優化結果（前7名，依超額報酬排序）===')
print(results_df.sort_values('OUT', ascending=False).head(7).to_string(index=False))

# ── 用最佳參數在 out-of-sample 驗證 ──────────────────────
best_row = results_df.sort_values('OUT', ascending=False).iloc[0]
best_sma1 = int(best_row['SMA1'])
best_sma2 = int(best_row['SMA2'])

data_out_result = run_strategy(data_out, best_sma1, best_sma2)
cumulative_out  = data_out_result[['Returns', 'Strategy']].cumsum().apply(np.exp)

fig, ax = plt.subplots(figsize=(12, 5))
cumulative_out['Returns'].plot(ax=ax, label='持有 PG（benchmark）')
cumulative_out['Strategy'].plot(ax=ax, label=f'SMA 策略（{best_sma1}/{best_sma2}）')
ax.set_title(f'Out-of-Sample 驗證（SMA1={best_sma1}, SMA2={best_sma2}）')
ax.legend()
plt.tight_layout()
plt.savefig('pg_sma_oos.png', dpi=150)
plt.show()

final_oos = np.exp(data_out_result[['Returns', 'Strategy']].sum())
print(f'\n=== Out-of-Sample 績效（SMA1={best_sma1}, SMA2={best_sma2}）===')
print(final_oos.round(4))
