# ── 套件匯入 ──────────────────────────────────────────────────────────────────
# 注意：執行前請確認已安裝 ib_insync
# 安裝指令：pip install ib_insync
# 同時需要 IBKR TWS 或 IB Gateway 在本機運行，並開啟 API 連線

import numpy as np
import pandas as pd
import pickle
import time
import logging
from datetime import datetime

# ib_insync：IBKR API 的 Python 封裝套件
# 功能對應書中 fxcmpy：連線、取得行情、下單、查倉位
from ib_insync import IB, Stock, MarketOrder, util

# 機器學習套件（載入已訓練好的模型）
from sklearn.svm import SVC

# ════════════════════════════════════════════════════════════════════════════════
# 第一步：訓練並儲存 SVM 模型
# 這段程式碼只需執行一次，之後直接載入 algorithm.pkl 即可
# ════════════════════════════════════════════════════════════════════════════════
import yfinance as yf

def train_and_save_model(symbol='PG', lags=5, train_frac=0.8,
                          filepath='algorithm.pkl'):
    """
    用歷史資料訓練 SVM 模型並儲存到檔案。
    """
    raw  = yf.download(symbol, start='2015-01-01', end='2024-12-31',
                       auto_adjust=True)
    data = pd.DataFrame(raw['Close'])
    data.columns = [symbol]
    data['returns']   = np.log(data[symbol] / data[symbol].shift(1))
    data.dropna(inplace=True)
    data['direction'] = np.sign(data['returns']).astype(int)

    # 建立二元化滯後特徵
    cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)
    data[cols] = np.where(data[cols] > 0, 1, 0)

    # 用前 train_frac 的資料訓練
    split = int(len(data) * train_frac)
    train = data.iloc[:split]

    model = SVC(C=1, kernel='linear', gamma='auto')
    model.fit(train[cols], train['direction'])

    # 用 pickle 儲存模型，之後可以直接載入使用
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'lags': lags, 'cols': cols}, f)

    print(f'模型已儲存至 {filepath}')
    return model, cols

# 執行訓練（第一次執行時取消以下的注解）
# model, cols = train_and_save_model()

# ════════════════════════════════════════════════════════════════════════════════
# 第二步：設定 logging（紀錄交易日誌）
# ════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),   # 寫入檔案
        logging.StreamHandler()                   # 同時印在終端機
    ]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# 第三步：IBKR 連線與交易參數設定
# ════════════════════════════════════════════════════════════════════════════════

# IBKR 連線設定
HOST     = '127.0.0.1'  # TWS 或 IB Gateway 的 IP（本機）
PORT     = 7497          # TWS Paper Trading 的預設 port（正式帳號用 7496）
CLIENT_ID = 1            # 每個連線需要不同的 client ID

# 交易參數
SYMBOL      = 'PG'       # 股票代碼
EXCHANGE    = 'SMART'    # IBKR 智慧路由
CURRENCY    = 'USD'
QUANTITY    = 10         # 每次交易的股數
BAR_SIZE    = '1 min'    # 重新取樣的 bar 長度（測試用1分鐘，實際部署可改5分鐘）
LAGS        = 5          # 特徵滯後天數

# ════════════════════════════════════════════════════════════════════════════════
# 第四步：定義 Online Algorithm 的核心邏輯
# ════════════════════════════════════════════════════════════════════════════════

class IBKROnlineTrader:
    """
    IBKR 即時交易策略類別。
    概念：
    1. 訂閱 PG 的即時 tick 資料
    2. 每累積足夠的 tick 後，重新取樣成固定長度的 bar
    3. 用最近5根 bar 的漲跌方向作為特徵，輸入 SVM 模型
    4. 根據預測方向和當前倉位決定是否下單
    """

    def __init__(self, model, cols, lags=5):
        self.model    = model      # 已訓練的 SVM 模型
        self.cols     = cols       # 特徵欄位名稱
        self.lags     = lags       # 滯後天數
        self.position = 0          # 當前倉位（+1 多、-1 空、0 中性）
        self.tick_df  = pd.DataFrame()  # 暫存 tick 資料
        self.ib       = IB()       # IBKR 連線物件

    def connect(self):
        """連線到 TWS 或 IB Gateway"""
        self.ib.connect(HOST, PORT, clientId=CLIENT_ID)
        logger.info(f'已連線到 IBKR（{HOST}:{PORT}）')

    def disconnect(self):
        """斷線"""
        self.ib.disconnect()
        logger.info('已斷線')

    def get_contract(self):
        """建立 PG 股票合約物件"""
        return Stock(SYMBOL, EXCHANGE, CURRENCY)

    def process_tick(self, ticker):
        """
        處理每個進來的 tick 資料。
        ticker：IBKR 提供的即時行情物件，包含 bid、ask、last 等欄位
        """
        # 取得當前中間價（mid price = (bid + ask) / 2）
        mid = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None
        if mid is None:
            return

        # 將 tick 資料加入暫存 DataFrame
        now = pd.Timestamp.now()
        self.tick_df = pd.concat([
            self.tick_df,
            pd.DataFrame({'mid': [mid]}, index=[now])
        ])

        # 重新取樣成 bar（例如每1分鐘一根）
        bar_df = self.tick_df['mid'].resample(BAR_SIZE).last().ffill()

        # 至少需要 lags+1 根 bar 才能計算特徵
        if len(bar_df) <= self.lags:
            return

        # 計算最近幾根 bar 的 log return 並轉為二元方向
        bar_df = bar_df.to_frame()
        bar_df['returns']   = np.log(bar_df['mid'] / bar_df['mid'].shift(1))
        bar_df['direction'] = np.where(bar_df['returns'] > 0, 1, 0)
        bar_df.dropna(inplace=True)

        # 取最近 lags 根 bar 的方向作為特徵
        features = bar_df['direction'].iloc[-self.lags:].values.reshape(1, -1)

        if features.shape[1] < self.lags:
            return

        # SVM 預測：+1 預測上漲，-1 預測下跌
        signal = self.modelpredict.(features)[0]

        logger.info(f'特徵：{features}，當前倉位：{self.position}，'
                    f'預測訊號：{signal}')

        # 根據訊號和當前倉位決定下單
        self._execute_order(signal)

    def _execute_order(self, signal):
        """
        根據訊號執行下單。
        邏輯：
        - 目前空/中性 且 訊號做多 → 買進
        - 目前多/中性 且 訊號做空 → 賣出（放空）
        - 書中沒有做空股票（stock 不像 FX 可以直接做空），這裡改為：
          多頭反轉時先平倉再做空（需要融券資格，Paper Trading 帳號可測試）
        """
        contract = self.get_contract()

        if self.position in [0, -1] and signal == 1:
            # 買進：數量 = QUANTITY + 當前空頭數量（若有）
            qty = QUANTITY + (QUANTITY if self.position == -1 else 0)
            order = MarketOrder('BUY', qty)
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # 等待成交
            self.position = 1
            logger.info(f'下單 BUY {qty} 股 PG，當前倉位：+1（多頭）')

        elif self.position in [0, 1] and signal == -1:
            # 賣出：先平多頭再建空頭
            qty = QUANTITY + (QUANTITY if self.position == 1 else 0)
            order = MarketOrder('SELL', qty)
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            self.position = -1
            logger.info(f'下單 SELL {qty} 股 PG，當前倉位：-1（空頭）')

        else:
            logger.info('無需換手，維持當前倉位')

    def close_all(self):
        """平掉所有倉位"""
        contract = self.get_contract()
        if self.position == 1:
            order = MarketOrder('SELL', QUANTITY)
            self.ib.placeOrder(contract, order)
            logger.info('平倉：賣出 PG 多頭')
        elif self.position == -1:
            order = MarketOrder('BUY', QUANTITY)
            self.ib.placeOrder(contract, order)
            logger.info('平倉：買回 PG 空頭')
        self.position = 0

    def run(self, duration_seconds=3600):
        """
        啟動即時交易，持續 duration_seconds 秒後自動平倉停止。
        """
        self.connect()
        contract  = self.get_contract()

        # 訂閱即時行情（reqMktData）
        ticker = self.ib.reqMktData(contract, '', False, False)

        logger.info(f'開始交易 PG，持續 {duration_seconds} 秒...')
        start_time = time.time()

        # 主迴圈：每隔一段時間處理最新 tick
        while time.time() - start_time < duration_seconds:
            self.ib.sleep(15)       # 每15秒處理一次
            self.process_tick(ticker)

        # 時間到：平倉並斷線
        logger.info('交易時間結束，開始平倉...')
        self.close_all()
        self.ib.cancelMktData(contract)
        self.disconnect()


# ════════════════════════════════════════════════════════════════════════════════
# 執行入口
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 載入已訓練的模型
    with open('algorithm.pkl', 'rb') as f:
        saved = pickle.load(f)

    model = saved['model']
    cols  = saved['cols']
    lags  = saved['lags']

    # 建立交易者物件並啟動
    trader = IBKROnlineTrader(model=model, cols=cols, lags=lags)

    # 注意：以下指令會真的連線到 IBKR 並下單
    # 請先在 Paper Trading 帳號測試，確認邏輯正確後再切換正式帳號
    # trader.run(duration_seconds=3600)  # 執行1小時

    print('概念7 程式碼已載入完成。')
    print('執行前請確認：')
    print('  1. IBKR TWS 或 IB Gateway 已開啟')
    print('  2. API 連線已在 TWS 設定中啟用')
    print('  3. algorithm.pkl 檔案存在（先執行 train_and_save_model()）')
    print('  4. 先用 Paper Trading 帳號測試（PORT=7497）')
