# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ── 參數設定 ──────────────────────────────────────────────────────────────────
SYMBOL = 'PG'
START  = '2010-01-01'
END    = '2024-12-31'
LAGS   = 5  # 使用前5天的 log return 作為特徵

# ── 資料準備 ──────────────────────────────────────────────────────────────────
raw  = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['returns']   = np.log(data[SYMBOL] / data[SYMBOL].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)

# 建立5個滯後特徵
cols = []
for lag in range(1, LAGS + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

# ── 特徵離散化函式 ────────────────────────────────────────────────────────────
def create_bins(data, cols, bins=[0]):
    """將連續特徵轉為離散值，bins=[0] 表示以0為界分成0和1兩類"""
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)
    return cols_bin

# ── 計算離散化用的 bins（用平均值和標準差切出4個區間）────────────────────────
mu   = data['returns'].mean()
std  = data['returns'].std()
bins = [mu - std, mu, mu + std]  # 將報酬分成4個區間

cols_bin = create_bins(data, cols, bins=bins)

# ════════════════════════════════════════════════════════════════════════════════
# 定義所有分類模型
# ════════════════════════════════════════════════════════════════════════════════
C = 1  # 正則化參數（控制模型複雜度，越小越正則化）

models = {
    'log_reg':  linear_model.LogisticRegression(C=C, max_iter=1000),
    'gauss_nb': GaussianNB(),
    'svm':      SVC(C=C, kernel='linear', gamma='auto')
}

def fit_models(train_data):
    """用訓練資料訓練所有模型"""
    for name, model in models.items():
        model.fit(train_data[cols_bin], train_data['direction'])

def derive_positions(test_data):
    """用已訓練模型對測試資料產生倉位預測"""
    for name, model in models.items():
        test_data[f'pos_{name}'] = model.predict(test_data[cols_bin])

def evaluate_strats(test_data):
    """計算每個模型的策略報酬"""
    sel = ['returns']
    for name in models.keys():
        col = f'strat_{name}'
        test_data[col] = test_data[f'pos_{name}'] * test_data['returns']
        sel.append(col)
    return sel

# ════════════════════════════════════════════════════════════════════════════════
# 第一部分：Sequential Train-Test Split（時序分割）
# 概念：前70%資料訓練，後30%資料測試，模擬真實交易情境
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

print('=== Sequential Split 測試集績效 ===')
print(test_seq[sel_seq].sum().apply(np.exp).round(4))

print(data['direction'].value_counts())
print(test_seq['pos_svm'].value_counts())

print('\n=== Sequential Split 預測準確率 ===')
for name in models.keys():
    acc = (test_seq['direction'] == test_seq[f'pos_{name}']).mean()
    print(f'  {name}: {acc:.2%}')

fig, ax = plt.subplots(figsize=(12, 5))
test_seq[sel_seq].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG 分類策略累積績效（Sequential Split）')
ax.legend(['持有 PG'] + list(models.keys()))
plt.tight_layout()
plt.savefig('pg_cls_sequential.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# 第二部分：Randomized Train-Test Split（隨機分割）
# 概念：假設歷史價格規律不隨時間改變，隨機抽取訓練和測試資料
# 注意：這個假設在金融資料中不一定成立
# ════════════════════════════════════════════════════════════════════════════════
train_rand, test_rand = train_test_split(
    data, test_size=0.5, shuffle=True, random_state=100)

train_rand = train_rand.copy().sort_index()
test_rand  = test_rand.copy().sort_index()

fit_models(train_rand)
derive_positions(test_rand)
sel_rand = evaluate_strats(test_rand)

print('\n=== Randomized Split 測試集績效 ===')
print(test_rand[sel_rand].sum().apply(np.exp).round(4))

fig, ax = plt.subplots(figsize=(12, 5))
test_rand[sel_rand].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG 分類策略累積績效（Randomized Split）')
ax.legend(['持有 PG'] + list(models.keys()))
plt.tight_layout()
plt.savefig('pg_cls_randomized.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# 第三部分：Deep Neural Network（深度神經網路）
# 概念：MLPClassifier 是多層感知器，hidden_layer_sizes 決定隱藏層的結構
# 例如 [256, 256] 表示兩個隱藏層，每層256個神經元
# ════════════════════════════════════════════════════════════════════════════════
dnn_model = MLPClassifier(
    solver='adam',              # 優化器，adam 適合大型資料
    alpha=1e-4,                 # L2 正則化強度，防止過擬合
    hidden_layer_sizes=[256, 256],  # 兩層隱藏層，各256個神經元
    max_iter=500,               # 最大訓練迭代次數
    random_state=1
)

# 用 sequential split 的訓練集訓練 DNN
dnn_model.fit(train_seq[cols_bin], train_seq['direction'])

# 在測試集上預測
test_seq['pos_dnn'] = dnn_model.predict(test_seq[cols_bin])
test_seq['strat_dnn'] = test_seq['pos_dnn'] * test_seq['returns']

dnn_acc  = (test_seq['direction'] == test_seq['pos_dnn']).mean()
dnn_perf = test_seq[['returns', 'strat_dnn']].sum().apply(np.exp)

print('\n=== DNN 策略績效（Sequential Split）===')
print(dnn_perf.round(4))
print(f'DNN 預測準確率：{dnn_acc:.2%}')

fig, ax = plt.subplots(figsize=(12, 5))
test_seq[['returns', 'strat_dnn']].cumsum().apply(np.exp).plot(ax=ax)
ax.set_title('PG DNN 策略累積績效（Sequential Split）')
ax.legend(['持有 PG（benchmark）', 'DNN 策略'])
plt.tight_layout()
plt.savefig('pg_dnn_performance.png', dpi=150)
plt.show()
