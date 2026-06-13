"""
PG 寶僑公司股價歷史 Time Series 分析
使用 yfinance 抓取最早可取得的資料至今日
"""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


# 設定資料與圖片的儲存路徑
PG_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VOO_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PG_CSV_PATH = os.path.join(PG_DATA_DIR, "pg_price_history.csv")
VOO_CSV_PATH = os.path.join(VOO_DATA_DIR, "voo_price_history.csv")
PNG_PATH = os.path.join(os.path.dirname(__file__), "pg_vs_voo_time_series.png")


# 確保 data 資料夾存在
os.makedirs(PG_DATA_DIR, exist_ok=True)
os.makedirs(VOO_DATA_DIR, exist_ok=True)

# 使用 yfinance 抓取 PG 從最早可取得日期到今天的日線收盤價
print("正在下載 PG 股價歷史資料...")
ticker = yf.Ticker("PG")
pg_df = ticker.history(period="max")

# 使用 yfinance 抓取 VOO 從最早可取得日期到今天的日線收盤價
print("正在下載 VOO 股價歷史資料...")
ticker = yf.Ticker("VOO")
voo_df = ticker.history(period="max")

# 只保留收盤價欄位，並去除時區資訊（避免 CSV 相容性問題）
pg_df.index = pg_df.index.tz_localize(None)
close_pg_df = pg_df[["Close"]].copy()
close_pg_df.index.name = "Date"

# 只保留收盤價欄位，並去除時區資訊（避免 CSV 相容性問題）
voo_df.index = voo_df.index.tz_localize(None)
close_voo_df = voo_df[["Close"]].copy()
close_voo_df.index.name = "Date"
start_date = close_voo_df.index.min()
close_pg_df = close_pg_df[close_pg_df.index >= start_date]

# Modify the data as the stock prices devided by the initial stock price at the time when VOO was created.
close_pg_df["Close"] = close_pg_df["Close"] / close_pg_df["Close"].iloc[0] * 100
close_voo_df["Close"] = close_voo_df["Close"] / close_voo_df["Close"].iloc[0] * 100

# 將資料存成 CSV 檔案
close_pg_df.to_csv(PG_CSV_PATH)
print(f"股價資料已儲存至：{PG_CSV_PATH}（共 {len(close_pg_df)} 筆）")

# 將資料存成 CSV 檔案
close_voo_df.to_csv(VOO_CSV_PATH)
print(f"股價資料已儲存至：{VOO_CSV_PATH}（共 {len(close_voo_df)} 筆）")

# 繪製 time series 折線圖
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(close_pg_df.index, close_pg_df["Close"], linewidth=0.8, color="#1f77b4", label="PG")
ax.plot(close_voo_df.index, close_voo_df["Close"], linewidth=0.8, color="#b41f1f", label="VOO")

# 設定標題與軸標籤
ax.set_title("PG versus VOO Stock Price History", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Normalized Closing Price\n(Base = 100, Starting from VOO IPO Date)", fontsize=12)

# 設定 x 軸日期格式，每 1 年顯示一個主刻度
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.YearLocator(2))
plt.xticks(rotation=45)

# 加上網格線，方便閱讀
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
ax.legend(fontsize=11)

# 顯示資料起訖日期範圍於圖表右下角
start_date = close_pg_df.index.min().strftime("%Y-%m-%d")
end_date = close_pg_df.index.max().strftime("%Y-%m-%d")
ax.annotate(
    f"{start_date} ~ {end_date}",
    xy=(1, 0),
    xycoords="axes fraction",
    fontsize=9,
    color="gray",
    ha="right",
    va="bottom",
)

plt.tight_layout()

# 將圖存成 PNG 檔案
fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
print(f"圖表已儲存至：{PNG_PATH}")

plt.close(fig)
print("分析完成。")
