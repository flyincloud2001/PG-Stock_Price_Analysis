# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ── 參數設定 ──────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'

# ════════════════════════════════════════════════════════════════════════════════
# 第一部分：Kelly Criterion 在二元賭局中的模擬
# 概念：在一個勝率 p > 0.5 的賭局中，每局該押多少比例的資金才能最大化長期財富？
# 公式：f* = p - q = p - (1 - p) = 2p - 1
# ════════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

p = 0.55        # 勝率 55%
f_star = p - (1 - p)  # Kelly 最優比例 = 0.10
I = 50          # 模擬 50 條路徑
n = 200         # 每條路徑 200 次試驗

def run_simulation(f, p=0.55, n=200, I=50):
    """
    模擬二元賭局。
    f: 每局投入的資金比例
    回傳形狀為 (n, I) 的矩陣，每欄為一條模擬路徑
    """
    c = np.zeros((n, I))
    c[0] = 100  # 初始資金 100 元
    for i in range(I):
        for t in range(1, n):
            outcome = np.random.binomial(1, p)  # 1=贏，0=輸
            if outcome > 0:
                c[t, i] = (1 + f) * c[t - 1, i]  # 贏：資金增加
            else:
                c[t, i] = (1 - f) * c[t - 1, i]  # 輸：資金減少
    return c

# 用不同 f 值模擬
c_optimal   = run_simulation(f_star)        # f* = 0.10
c_half      = run_simulation(f_star * 0.5)  # 半 Kelly = 0.05
c_double    = run_simulation(f_star * 2.5)  # 過度下注 = 0.25
c_extreme   = run_simulation(0.5)           # 極端過度下注

# ── 圖1：最優 Kelly 的所有模擬路徑 ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(c_optimal, 'b', lw=0.3, alpha=0.5)
ax.plot(c_optimal.mean(axis=1), 'r', lw=2, label='平均資金曲線')
ax.set_title(f'二元賭局模擬（f* = {f_star:.2f}，勝率 = {p}）')
ax.set_xlabel('試驗次數')
ax.set_ylabel('資金')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kelly_simulation.png', dpi=150)
plt.show()

# ── 圖2：不同 f 值的平均資金曲線比較 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(c_optimal.mean(axis=1), label=f'f* = {f_star:.2f}（最優）')
ax.plot(c_half.mean(axis=1),    label=f'f = {f_star*0.5:.2f}（半 Kelly）')
ax.plot(c_double.mean(axis=1),  label=f'f = {f_star*2.5:.2f}（過度下注）')
ax.plot(c_extreme.mean(axis=1), label='f = 0.50（極端）')
ax.set_title('不同 f 值下的平均資金曲線')
ax.set_xlabel('試驗次數')
ax.set_ylabel('平均資金')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kelly_comparison.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# 第二部分：Kelly Criterion 應用到 PG 股票
# 公式：f* = (μ - r) / σ²
# μ：年化預期報酬率，r：無風險利率，σ²：年化報酬變異數
# ════════════════════════════════════════════════════════════════════════════════
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['returns'] = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)

# 計算年化統計量（252 個交易日）
mu    = data['returns'].mean() * 252       # 年化平均報酬
sigma = data['returns'].std() * np.sqrt(252)  # 年化波動率
r     = 0.0                                # 簡化：無風險利率設為 0

# Kelly 最優槓桿比例
f_pg = (mu - r) / sigma ** 2

print(f'=== PG Kelly Criterion ===')
print(f'年化平均報酬 μ：{mu:.4f}')
print(f'年化波動率 σ：{sigma:.4f}')
print(f'最優 Kelly 槓桿 f*：{f_pg:.4f}')
print(f'半 Kelly 槓桿：{f_pg * 0.5:.4f}')

# ── 模擬 Kelly 槓桿策略下的資金曲線 ──────────────────────────────────────────
def kelly_strategy(data, symbol, f):
    """
    模擬 Kelly 槓桿策略。
    每天根據當日報酬調整資本部位，再根據新資本重設槓桿。
    equ：自有資本（equity）
    cap：實際投入資本（capital = equ * f）
    """
    equ_col = f'equity_{f:.2f}'
    cap_col = f'capital_{f:.2f}'
    data[equ_col] = 1.0   # 初始自有資本設為 1
    data[cap_col] = f      # 初始投入資本 = 1 * f

    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]
        # 新資本 = 舊資本 × exp(當日報酬)
        data.loc[t, cap_col] = (data.loc[t_1, cap_col] *
                                 math.exp(data.loc[t, 'returns']))
        # 新資本 = 舊資本 + 資本變動（損益）
        data.loc[t, equ_col] = (data.loc[t_1, equ_col] +
                                  data.loc[t, cap_col] -
                                  data.loc[t_1, cap_col])
        # 根據新自有資本重設投入資本
        data.loc[t, cap_col] = data.loc[t, equ_col] * f

    return equ_col

eq_full = kelly_strategy(data, SYMBOL, f_pg)        # 全 Kelly
eq_half = kelly_strategy(data, SYMBOL, f_pg * 0.5)  # 半 Kelly
eq_qtr  = kelly_strategy(data, SYMBOL, f_pg * 0.25) # 四分之一 Kelly

# ── 圖3：PG Kelly 策略 vs 直接持有 ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data['returns'].cumsum().apply(np.exp).plot(ax=ax, label='直接持有 PG')
data[[eq_full, eq_half, eq_qtr]].plot(ax=ax,
    label=[f'全 Kelly (f={f_pg:.2f})',
           f'半 Kelly (f={f_pg*0.5:.2f})',
           f'四分之一 Kelly (f={f_pg*0.25:.2f})'])
ax.set_title(f'PG Kelly 槓桿策略 vs 直接持有')
ax.legend()
plt.tight_layout()
plt.savefig('pg_kelly_equity.png', dpi=150)
plt.show()
