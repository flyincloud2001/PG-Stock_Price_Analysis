# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ── 參數設定 ──────────────────────────────────────────────────────────────────
SYMBOL = 'PG'          # 寶僑公司股票代碼
START  = '2010-01-01'  # 資料起始日
END    = '2024-12-31'  # 資料結束日
LAGS   = 5             # 滯後天數（使用過去5天的股價來預測今天）

# ── 抓取資料 ──────────────────────────────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data.dropna(inplace=True)

# ── 建立滯後欄位 ───────────────────────────────────────────────────────────────
# 概念：若 RWH 成立，lag_1（昨天股價）的係數應接近 1，其餘滯後接近 0
# 這代表「今天最好的預測值就是昨天的價格」，技術分析因此沒有意義
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data[SYMBOL].shift(lag)  # 將股價向後平移 lag 天
    cols.append(col)

data.dropna(inplace=True)

# ── OLS 回歸：用過去5天股價預測今天股價 ──────────────────────────────────────
# np.linalg.lstsq 解最小平方問題：找出最佳回歸係數 beta
# 公式：price_today ≈ beta_0 + beta_1*lag1 + beta_2*lag2 + ... + beta_5*lag5
X = np.column_stack([np.ones(len(data)), data[cols].values])  # 加入截距項
y = data[SYMBOL].values

# lstsq 回傳 (係數, 殘差, rank, 奇異值)，我們只取係數
beta = np.linalg.lstsq(X, y, rcond=None)[0]

# ── 圖1：回歸係數長條圖 ────────────────────────────────────────────────────────
# 若 RWH 成立，lag_1 的係數接近 1，其餘接近 0
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(['intercept'] + cols, beta, color='steelblue')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('PG OLS 回歸係數（用過去5天股價預測今天股價）')
ax.set_xlabel('滯後項')
ax.set_ylabel('係數值')
plt.tight_layout()
plt.savefig('pg_rwh_coefficients.png', dpi=150)
plt.show()

# ── 計算預測值 ────────────────────────────────────────────────────────────────
# 將回歸係數套用回資料，得到每天的預測股價
data['prediction'] = X @ beta  # 矩陣乘法：X 乘以 beta 向量

# ── 圖2：實際股價 vs 預測股價 ─────────────────────────────────────────────────
# 若預測線幾乎等於實際線向右平移一天，代表 lag_1 主導，支持 RWH
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data[SYMBOL],    label='PG 實際股價',  linewidth=1.2)
ax.plot(data.index, data['prediction'], label='OLS 預測股價', linewidth=1.0,
        linestyle='--', alpha=0.8)
ax.set_title('PG 實際股價 vs OLS 預測股價')
ax.legend()
plt.tight_layout()
plt.savefig('pg_rwh_prediction.png', dpi=150)
plt.show()

# ── 印出回歸係數數值 ──────────────────────────────────────────────────────────
print('=== OLS 回歸係數 ===')
labels = ['截距'] + [f'lag_{i}' for i in range(1, LAGS + 1)]
for label, coef in zip(labels, beta):
    print(f'  {label:>10s}：{coef:.6f}')

# ── 結論說明 ──────────────────────────────────────────────────────────────────
# 若 lag_1 係數接近 1，其餘接近 0：
#   → 支持 RWH：今天最好的預測就是昨天的價格，過去資訊無額外預測力
#   → 技術指標（SMA、RSI 等）理論上不應帶來超額報酬
# 若其他滯後項係數明顯不為 0：
#   → 弱式效率市場假說可能不完全成立，存在可利用的價格規律
print('\n=== 解讀 ===')
print(f'lag_1 係數：{beta[1]:.6f}（接近 1 代表支持 RWH）')
print(f'lag_2 到 lag_5 係數平均絕對值：'
      f'{np.mean(np.abs(beta[2:])):.6f}（接近 0 代表支持 RWH）')