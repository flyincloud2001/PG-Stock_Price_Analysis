import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# ── 頁面標題 ──────────────────────────────────────────────
st.title("股票績效分析工具")

# ── 側邊欄：使用者輸入區 ──────────────────────────────────
st.sidebar.header("參數設定")

# 主要股票代號輸入
main_ticker = st.sidebar.text_input("主要股票代號", value="PG")

# 基準股票代號輸入
benchmark_ticker = st.sidebar.text_input("基準股票代號", value="VOO")

# 開始日期輸入
start_date = st.sidebar.date_input("開始日期", value=date(2010, 1, 1), min_value=date(1962, 1, 2), max_value=date.today())

# 結束日期輸入（預設今天）
end_date = st.sidebar.date_input("結束日期", value=date.today())

# 執行分析按鈕
run_button = st.sidebar.button("Run Analysis")

# ── 按下按鈕後執行分析 ────────────────────────────────────
if run_button:
    # 驗證日期順序
    if start_date >= end_date:
        st.error("開始日期必須早於結束日期，請重新設定。")
        st.stop()

    # 從 yfinance 抓取兩支股票的歷史收盤價
    with st.spinner("下載股價資料中..."):
        raw = yf.download(
            [main_ticker, benchmark_ticker],
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

    # 取出收盤價欄位；若只有單一股票 yfinance 回傳 DataFrame 而非 MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][[main_ticker, benchmark_ticker]].dropna()
    else:
        close = raw[["Close"]].rename(columns={"Close": main_ticker}).dropna()

    # 檢查資料是否足夠
    if close.empty or len(close) < 2:
        st.error("無法取得足夠的股價資料，請確認股票代號與日期範圍。")
        st.stop()

    # 取期初與期末收盤價
    price_start_main = close[main_ticker].iloc[0]
    price_end_main = close[main_ticker].iloc[-1]

    # 計算實際天數（以年為單位）
    days = (close.index[-1] - close.index[0]).days
    years = days / 365

    # 計算 Total Return（總報酬率）= (期末價 - 期初價) / 期初價 × 100
    total_return = (price_end_main - price_start_main) / price_start_main * 100

    # 計算 CAGR（年化複合成長率）= (期末價 / 期初價)^(1/年數) - 1
    cagr = (price_end_main / price_start_main) ** (1 / years) - 1

    # 計算兩支股票每日對數報酬率
    log_returns = np.log(close / close.shift(1)).dropna()

    # 計算兩支股票對數報酬率的相關係數
    correlation = log_returns[main_ticker].corr(log_returns[benchmark_ticker])

    # ── 顯示統計數據 ─────────────────────────────────────
    st.subheader(f"{main_ticker} 績效指標（{start_date} ～ {end_date}）")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{total_return:.2f}%")
    col2.metric("CAGR", f"{cagr * 100:.2f}%")
    col3.metric(f"與 {benchmark_ticker} 相關係數", f"{correlation:.4f}")

    # ── 繪製標準化股價折線圖（起點皆為 100）────────────────
    st.subheader("標準化股價走勢（基準 = 100）")

    # 將兩支股票股價標準化，使起點皆為 100
    normalized = close / close.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(normalized.index, normalized[main_ticker], label=main_ticker, linewidth=1.8)
    ax.plot(
        normalized.index,
        normalized[benchmark_ticker],
        label=benchmark_ticker,
        linewidth=1.8,
        linestyle="--",
    )

    ax.set_title(f"{main_ticker} vs {benchmark_ticker} 標準化股價（起點 = 100）")
    ax.set_xlabel("日期")
    ax.set_ylabel("標準化股價")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 將圖表顯示在 Streamlit 頁面上
    st.pyplot(fig)
    plt.close(fig)
