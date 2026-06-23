# ── Imports ───────────────────────────────────────────────────────────────────
# Note: install ib_insync before running:
#   pip install ib_insync
# IBKR TWS or IB Gateway must be running locally with API connections enabled.

import numpy as np
import pandas as pd
import pickle
import time
import logging
from datetime import datetime

# ib_insync: Python wrapper for the IBKR API
# Provides the same functions as fxcmpy in the book: connect, get quotes, place orders, query positions
from ib_insync import IB, Stock, MarketOrder, util

# ML package (load a pre-trained model)
from sklearn.svm import SVC

# ════════════════════════════════════════════════════════════════════════════════
# Step 1: Train and save the SVM model
# Run this section only once; afterwards load algorithm.pkl directly.
# ════════════════════════════════════════════════════════════════════════════════
import yfinance as yf

def train_and_save_model(symbol='PG', lags=5, train_frac=0.8,
                          filepath='algorithm.pkl'):
    """
    Train an SVM model on historical data and save it to a file.
    """
    raw  = yf.download(symbol, start='2015-01-01', end='2024-12-31',
                       auto_adjust=True)
    data = pd.DataFrame(raw['Close'])
    data.columns = [symbol]
    data['returns']   = np.log(data[symbol] / data[symbol].shift(1))
    data.dropna(inplace=True)
    data['direction'] = np.sign(data['returns']).astype(int)

    # Build binary lag features
    cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)
    data[cols] = np.where(data[cols] > 0, 1, 0)

    # Train on the first train_frac of the data
    split = int(len(data) * train_frac)
    train = data.iloc[:split]

    model = SVC(C=1, kernel='linear', gamma='auto')
    model.fit(train[cols], train['direction'])

    # Serialize the model with pickle for later use
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'lags': lags, 'cols': cols}, f)

    print(f'Model saved to {filepath}')
    return model, cols

# Run training (uncomment the line below on first run)
# model, cols = train_and_save_model()

# ════════════════════════════════════════════════════════════════════════════════
# Step 2: Configure logging (record trade log)
# ════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),   # Write to file
        logging.StreamHandler()                   # Also print to terminal
    ]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# Step 3: IBKR connection and trading parameters
# ════════════════════════════════════════════════════════════════════════════════

# IBKR connection settings
HOST     = '127.0.0.1'  # IP of TWS or IB Gateway (localhost)
PORT     = 7497          # Default port for TWS Paper Trading (live account uses 7496)
CLIENT_ID = 1            # Each connection requires a unique client ID

# Trading parameters
SYMBOL      = 'PG'       # Stock ticker
EXCHANGE    = 'SMART'    # IBKR smart order routing
CURRENCY    = 'USD'
QUANTITY    = 10         # Number of shares per trade
BAR_SIZE    = '1 min'    # Bar length for resampling (1 min for testing; use 5 min in production)
LAGS        = 5          # Number of lag features

# ════════════════════════════════════════════════════════════════════════════════
# Step 4: Core logic of the Online Algorithm
# ════════════════════════════════════════════════════════════════════════════════

class IBKROnlineTrader:
    """
    IBKR live trading strategy class.
    Logic:
    1. Subscribe to real-time tick data for PG.
    2. Resample accumulated ticks into fixed-length bars.
    3. Use the direction of the last 5 bars as features for the SVM model.
    4. Decide whether to place an order based on the predicted direction and current position.
    """

    def __init__(self, model, cols, lags=5):
        self.model    = model      # Pre-trained SVM model
        self.cols     = cols       # Feature column names
        self.lags     = lags       # Number of lag features
        self.position = 0          # Current position (+1 long, -1 short, 0 flat)
        self.tick_df  = pd.DataFrame()  # Buffer for tick data
        self.ib       = IB()       # IBKR connection object

    def connect(self):
        """Connect to TWS or IB Gateway."""
        self.ib.connect(HOST, PORT, clientId=CLIENT_ID)
        logger.info(f'Connected to IBKR ({HOST}:{PORT})')

    def disconnect(self):
        """Disconnect from IBKR."""
        self.ib.disconnect()
        logger.info('Disconnected')

    def get_contract(self):
        """Create the PG stock contract object."""
        return Stock(SYMBOL, EXCHANGE, CURRENCY)

    def process_tick(self, ticker):
        """
        Process each incoming tick.
        ticker: IBKR real-time quote object containing bid, ask, last, etc.
        """
        # Compute the mid price: (bid + ask) / 2
        mid = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None
        if mid is None:
            return

        # Append tick to the buffer DataFrame
        now = pd.Timestamp.now()
        self.tick_df = pd.concat([
            self.tick_df,
            pd.DataFrame({'mid': [mid]}, index=[now])
        ])

        # Resample ticks into bars (e.g., 1-minute bars)
        bar_df = self.tick_df['mid'].resample(BAR_SIZE).last().ffill()

        # Need at least lags+1 bars to compute features
        if len(bar_df) <= self.lags:
            return

        # Compute log returns for each bar and convert to binary direction
        bar_df = bar_df.to_frame()
        bar_df['returns']   = np.log(bar_df['mid'] / bar_df['mid'].shift(1))
        bar_df['direction'] = np.where(bar_df['returns'] > 0, 1, 0)
        bar_df.dropna(inplace=True)

        # Take the direction of the last `lags` bars as the feature vector
        features = bar_df['direction'].iloc[-self.lags:].values.reshape(1, -1)

        if features.shape[1] < self.lags:
            return

        # SVM prediction: +1 = predict up, -1 = predict down
        signal = self.model.predict(features)[0]

        logger.info(f'Features: {features}, current position: {self.position}, '
                    f'predicted signal: {signal}')

        # Place order based on signal and current position
        self._execute_order(signal)

    def _execute_order(self, signal):
        """
        Execute an order based on the signal.
        Logic:
        - Currently short/flat and signal is long  -> buy
        - Currently long/flat and signal is short  -> sell (go short)
        - Note: shorting equities (unlike FX) requires margin/short-selling approval.
          In this implementation, reversing a long position first closes it then goes short.
          Paper Trading accounts can be used to test this.
        """
        contract = self.get_contract()

        if self.position in [0, -1] and signal == 1:
            # Buy: quantity = QUANTITY + any existing short position
            qty = QUANTITY + (QUANTITY if self.position == -1 else 0)
            order = MarketOrder('BUY', qty)
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Wait for fill
            self.position = 1
            logger.info(f'Order BUY {qty} shares of PG; current position: +1 (long)')

        elif self.position in [0, 1] and signal == -1:
            # Sell: close long first, then go short
            qty = QUANTITY + (QUANTITY if self.position == 1 else 0)
            order = MarketOrder('SELL', qty)
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            self.position = -1
            logger.info(f'Order SELL {qty} shares of PG; current position: -1 (short)')

        else:
            logger.info('No position change needed; maintaining current position')

    def close_all(self):
        """Close all open positions."""
        contract = self.get_contract()
        if self.position == 1:
            order = MarketOrder('SELL', QUANTITY)
            self.ib.placeOrder(contract, order)
            logger.info('Closing position: selling PG long')
        elif self.position == -1:
            order = MarketOrder('BUY', QUANTITY)
            self.ib.placeOrder(contract, order)
            logger.info('Closing position: buying back PG short')
        self.position = 0

    def run(self, duration_seconds=3600):
        """
        Start live trading and automatically close positions after duration_seconds.
        """
        self.connect()
        contract  = self.get_contract()

        # Subscribe to real-time market data
        ticker = self.ib.reqMktData(contract, '', False, False)

        logger.info(f'Trading PG for {duration_seconds} seconds...')
        start_time = time.time()

        # Main loop: process the latest tick every 15 seconds
        while time.time() - start_time < duration_seconds:
            self.ib.sleep(15)
            self.process_tick(ticker)

        # Time up: close all positions and disconnect
        logger.info('Trading period ended; closing all positions...')
        self.close_all()
        self.ib.cancelMktData(contract)
        self.disconnect()


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Load pre-trained model
    with open('algorithm.pkl', 'rb') as f:
        saved = pickle.load(f)

    model = saved['model']
    cols  = saved['cols']
    lags  = saved['lags']

    # Instantiate the trader and start
    trader = IBKROnlineTrader(model=model, cols=cols, lags=lags)

    # WARNING: the line below will connect to IBKR and place real orders.
    # Test with a Paper Trading account first (PORT=7497) before switching to live (PORT=7496).
    # trader.run(duration_seconds=3600)  # Run for 1 hour

    print('Online algorithm script loaded successfully.')
    print('Before running, verify:')
    print('  1. IBKR TWS or IB Gateway is running')
    print('  2. API connections are enabled in TWS settings')
    print('  3. algorithm.pkl exists (run train_and_save_model() first)')
    print('  4. Test with Paper Trading account (PORT=7497)')