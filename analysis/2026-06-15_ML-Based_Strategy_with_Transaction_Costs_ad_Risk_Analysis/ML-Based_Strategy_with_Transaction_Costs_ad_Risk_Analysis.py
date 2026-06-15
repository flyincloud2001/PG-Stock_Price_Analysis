# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ── 參數設定 ──────────────────────────────────────────────────────────────────
SYMBOL     = 'PG'
START      = '2010-01-01'
END        = '2024-12-31'
LAGS       = 5       # 使用前5天的方向作為特徵
TRAIN_FRAC = 0.8     # 前80%資料作為訓練集
LEVERAGE   = 3       # 槓桿倍數（用於風險分析）
EQUITY     = 10000   # 初始自有資本（美元）

# ── 抓取資料 ──────────────────────────────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['returns']   = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)

# ── 建立二元化滯後特徵 ────────────────────────────────────────────────────────
# 特徵值：0 表示當天下跌，1 表示當天上漲
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

# 將連續 log return 轉為二元值（0/1）

# ── 估算比例交易成本 ──────────────────────────────────────────────────────────
# 書中用 FXCM 的 bid-ask spread，這裡用 PG 股票的典型估算值
# 一般股票的 bid-ask spread 約為股價的 0.01%~0.05%
# 此處設定為 0.02%（也就是每筆交易的單程成本）
ptc = 0.0002  # Proportional Transaction Cost（比例交易成本）

# ════════════════════════════════════════════════════════════════════════════════
# SVM 策略訓練與回測（含交易成本）
# ════════════════════════════════════════════════════════════════════════════════
split = int(len(data) * TRAIN_FRAC)
train = data.iloc[:split].copy()
test  = data.iloc[split:].copy()

# 訓練 SVM 分類器
model = SVC(C=1, kernel='linear', gamma='auto')
model.fit(train[cols], train['direction'])

# 在測試集上預測倉位
test['position'] = model.predict(test[cols])
print(test['position'].value_counts())

# 準確率（hit ratio）
train_acc = accuracy_score(train['direction'], model.predict(train[cols]))
test_acc  = accuracy_score(test['direction'], test['position'])
print(f'=== SVM 策略準確率 ===')
print(f'訓練集：{train_acc:.2%}')
print(f'測試集：{test_acc:.2%}')

# 無成本策略報酬
test['strategy'] = test['position'] * test['returns']

# 計算換手次數（倉位改變的次數）
n_trades = (test['position'].diff() != 0).sum()
print(f'\n換手次數：{n_trades}')

# 含成本策略報酬：每次換手扣除比例成本
test['strategy_tc'] = np.where(
    test['position'].diff() != 0,        # 有換手的那天
    test['strategy'] - ptc,              # 扣除交易成本
    test['strategy']                     # 無換手則不扣
)

# ── 總績效 ────────────────────────────────────────────────────────────────────
perf = test[['returns', 'strategy', 'strategy_tc']].sum().apply(np.exp)
print(f'\n=== 測試集總績效 ===')
print(perf.round(4))

# ── 圖1：累積績效比較 ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
test[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG SVM 策略累積績效（含/不含交易成本）')
ax.legend(['持有 PG（benchmark）', 'SVM 策略（無成本）', 'SVM 策略（含成本）'])
plt.tight_layout()
plt.savefig('pg_svm_performance.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# 最優槓桿計算（Kelly Criterion 應用到策略報酬）
# ════════════════════════════════════════════════════════════════════════════════
# 年化統計量（252 個交易日）
mean_ret  = test[['returns', 'strategy_tc']].mean() * 252
var_ret   = test[['returns', 'strategy_tc']].var() * 252

kelly_f   = mean_ret / var_ret
print(f'\n=== 最優 Kelly 槓桿 ===')
print(kelly_f.round(4))
print(f'\n半 Kelly：')
print((kelly_f * 0.5).round(4))

# ── 不同槓桿下的績效模擬 ─────────────────────────────────────────────────────
to_plot = ['returns', 'strategy_tc']
for lev in [1, 2, 3, 5]:
    col = f'strategy_tc_lev{lev}'
    test[col] = test['strategy_tc'] * lev
    to_plot.append(col)

fig, ax = plt.subplots(figsize=(12, 5))
test[to_plot].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG SVM 策略在不同槓桿下的累積績效')
ax.legend(['持有 PG', '策略（含成本，槓桿1）',
           '槓桿 1x', '槓桿 2x', '槓桿 3x', '槓桿 5x'])
plt.tight_layout()
plt.savefig('pg_svm_leverage.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# 風險分析
# ════════════════════════════════════════════════════════════════════════════════
# 使用槓桿 LEVERAGE 計算資金曲線
risk = pd.DataFrame()
risk['strategy_lev'] = test['strategy_tc'] * LEVERAGE

# 自有資本曲線：初始 EQUITY 元，依策略報酬累積
risk['equity'] = risk['strategy_lev'].cumsum().apply(np.exp) * EQUITY

# ── 最大回撤 ─────────────────────────────────────────────────────────────────
# 最大回撤 = 從歷史高點到當前低點的最大跌幅
risk['cummax']   = risk['equity'].cummax()          # 歷史最高資金
risk['drawdown'] = risk['cummax'] - risk['equity']  # 回撤金額

max_dd   = risk['drawdown'].max()
t_max_dd = risk['drawdown'].idxmax()

print(f'\n=== 風險分析（槓桿 {LEVERAGE}x，初始資金 ${EQUITY}）===')
print(f'最大回撤金額：${max_dd:.2f}')
print(f'最大回撤發生時間：{t_max_dd.date()}')

# 計算最長回撤期間（從一個高點到下一個高點的最長時間）
highs   = risk['drawdown'][risk['drawdown'] == 0]
print(highs)
if len(highs) > 1:
    periods = (highs.index[1:] - highs.index[:-1]).days
    max_period = periods.max()
    print(f'最長回撤天數：{max_period} 天')

# ── 圖2：資金曲線與最大回撤 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
risk[['equity', 'cummax']].plot(ax=ax, label=['資金曲線', '歷史最高'])
ax.axvline(t_max_dd, color='r', alpha=0.5, linestyle='--', label='最大回撤點')
ax.set_title(f'PG SVM 策略資金曲線（槓桿 {LEVERAGE}x）')
ax.legend()
plt.tight_layout()
plt.savefig('pg_svm_drawdown.png', dpi=150)
plt.show()

# ── 風險值 VaR（Value at Risk）────────────────────────────────────────────────
# VaR 代表在某個信心水準下，未來一段時間內最多可能損失多少
risk['equity_returns'] = np.log(risk['equity'] / risk['equity'].shift(1))
risk.dropna(inplace=True)

percs = np.array([0.01, 0.1, 1.0, 2.5, 5.0, 10.0])  # 百分位數
VaR   = scs.scoreatpercentile(EQUITY * risk['equity_returns'], percs)

print(f'\n=== VaR（日頻，信心水準 vs 最大損失）===')
print(f'{"信心水準":>12s} {"最大損失（$）":>15s}')
print('-' * 30)
for conf, var in zip(100 - percs, -VaR):
    print(f'{conf:>12.2f}% {var:>15.2f}')
