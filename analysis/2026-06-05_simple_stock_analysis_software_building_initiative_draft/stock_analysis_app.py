import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

# ── Page title ────────────────────────────────────────────
st.title("Stock Performance Analysis Tool")

# ── Sidebar: user input section ───────────────────────────
st.sidebar.header("Parameters")

# Primary stock ticker input
main_ticker = st.sidebar.text_input("Primary Ticker", value="PG")

# Benchmark stock ticker input
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="VOO")

# Start date input
start_date = st.sidebar.date_input("Start Date", value=date(2010, 1, 1), min_value=date(1962, 1, 2), max_value=date.today())

# End date input (defaults to today)
end_date = st.sidebar.date_input("End Date", value=date.today())

# Run analysis button
run_button = st.sidebar.button("Run Analysis")

# ── Execute analysis when button is clicked ───────────────
if run_button:
    # Validate date order
    if start_date >= end_date:
        st.error("Start date must be earlier than end date. Please adjust the date range.")
        st.stop()

    # Fetch historical closing prices for both tickers via yfinance
    with st.spinner("Downloading price data..."):
        raw = yf.download(
            [main_ticker, benchmark_ticker],
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )

    # Extract the Close column; yfinance returns a flat DataFrame (not MultiIndex) for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][[main_ticker, benchmark_ticker]].dropna()
    else:
        close = raw[["Close"]].rename(columns={"Close": main_ticker}).dropna()

    # Ensure there is enough data to calculate metrics
    if close.empty or len(close) < 2:
        st.error("Insufficient price data. Please verify the ticker symbols and date range.")
        st.stop()

    # First and last closing prices for the primary ticker
    price_start_main = close[main_ticker].iloc[0]
    price_end_main = close[main_ticker].iloc[-1]

    # Number of calendar days converted to years
    days = (close.index[-1] - close.index[0]).days
    years = days / 365

    # Total Return = (ending price - starting price) / starting price × 100
    total_return = (price_end_main - price_start_main) / price_start_main * 100

    # CAGR = (ending price / starting price)^(1 / years) - 1
    MIN_YEARS_FOR_CAGR = 1.0
    cagr = (price_end_main / price_start_main) ** (1 / years) - 1

    # Daily log returns for both tickers
    log_returns = np.log(close / close.shift(1)).dropna()

    # Pearson correlation coefficient of the two tickers' daily log returns
    correlation = log_returns[main_ticker].corr(log_returns[benchmark_ticker])

    # ── Display performance metrics ───────────────────────
    st.subheader(f"{main_ticker} Performance Metrics ({start_date} to {end_date})")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{total_return:.2f}%")
    if years >= MIN_YEARS_FOR_CAGR:
        col2.metric("CAGR", f"{cagr * 100:.2f}%")
    else:
        col2.metric("CAGR", "N/A (period < 1 year)")
    col3.metric(f"Correlation with {benchmark_ticker}", f"{correlation:.4f}")

    # ── Plot normalized price chart (both start at 100) ───
    st.subheader("Normalized Price Performance (Base = 100)")

    # Normalize both tickers so they start at 100
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

    ax.set_title(f"{main_ticker} vs {benchmark_ticker} Normalized Price (Base = 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Render the chart in the Streamlit page
    st.pyplot(fig)
    plt.close(fig)
