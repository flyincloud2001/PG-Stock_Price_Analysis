"""
PG (Procter & Gamble) Stock Price Historical Time Series Analysis
Fetching all available data from yfinance up to today
"""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Set paths for data and image storage
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "pg_price_history.csv")
PNG_PATH = os.path.join(os.path.dirname(__file__), "pg_time_series.png")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download PG closing price history from the earliest available date to today
print("Downloading PG stock price history...")
ticker = yf.Ticker("PG")
df = ticker.history(period="max")

# Keep only the Close column and remove timezone info (for CSV compatibility)
df.index = df.index.tz_localize(None)
close_df = df[["Close"]].copy()
close_df.index.name = "Date"

# Save data to CSV
close_df.to_csv(CSV_PATH)
print(f"Stock data saved to: {CSV_PATH} ({len(close_df)} records)")

# Plot the time series line chart
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(close_df.index, close_df["Close"], linewidth=0.8, color="#1f77b4")

# Set title and axis labels
ax.set_title("PG Stock Price History", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Close Price (USD)", fontsize=12)

# Set x-axis date format with major ticks every 10 years
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.YearLocator(2))
plt.xticks(rotation=45)

# Add grid lines for readability
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)

# Display date range in the bottom-right corner
start_date = close_df.index.min().strftime("%Y-%m-%d")
end_date = close_df.index.max().strftime("%Y-%m-%d")
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

# Save the chart as PNG
fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
print(f"Chart saved to: {PNG_PATH}")

plt.close(fig)
print("Analysis complete.")