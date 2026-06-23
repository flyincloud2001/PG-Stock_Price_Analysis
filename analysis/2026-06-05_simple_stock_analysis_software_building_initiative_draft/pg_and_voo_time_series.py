"""
PG (Procter & Gamble) Stock Price Historical Time Series Analysis
Fetching all available data from yfinance up to today
"""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


# Set paths for data and image storage
PG_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VOO_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PG_CSV_PATH = os.path.join(PG_DATA_DIR, "pg_price_history.csv")
VOO_CSV_PATH = os.path.join(VOO_DATA_DIR, "voo_price_history.csv")
PNG_PATH = os.path.join(os.path.dirname(__file__), "pg_vs_voo_time_series.png")


# Ensure the data directories exist
os.makedirs(PG_DATA_DIR, exist_ok=True)
os.makedirs(VOO_DATA_DIR, exist_ok=True)

# Download PG closing price history from the earliest available date to today
print("Downloading PG stock price history...")
ticker = yf.Ticker("PG")
pg_df = ticker.history(period="max")

# Download VOO closing price history from the earliest available date to today
print("Downloading VOO stock price history...")
ticker = yf.Ticker("VOO")
voo_df = ticker.history(period="max")

# Keep only the Close column and remove timezone info (for CSV compatibility)
pg_df.index = pg_df.index.tz_localize(None)
close_pg_df = pg_df[["Close"]].copy()
close_pg_df.index.name = "Date"

# Keep only the Close column and remove timezone info (for CSV compatibility)
voo_df.index = voo_df.index.tz_localize(None)
close_voo_df = voo_df[["Close"]].copy()
close_voo_df.index.name = "Date"
start_date = close_voo_df.index.min()
close_pg_df = close_pg_df[close_pg_df.index >= start_date]

# Normalize both series to 100 at the VOO IPO date
close_pg_df["Close"] = close_pg_df["Close"] / close_pg_df["Close"].iloc[0] * 100
close_voo_df["Close"] = close_voo_df["Close"] / close_voo_df["Close"].iloc[0] * 100

# Save data to CSV
close_pg_df.to_csv(PG_CSV_PATH)
print(f"PG stock data saved to: {PG_CSV_PATH} ({len(close_pg_df)} records)")

# Save data to CSV
close_voo_df.to_csv(VOO_CSV_PATH)
print(f"VOO stock data saved to: {VOO_CSV_PATH} ({len(close_voo_df)} records)")

# Plot the time series line chart
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(close_pg_df.index, close_pg_df["Close"], linewidth=0.8, color="#1f77b4", label="PG")
ax.plot(close_voo_df.index, close_voo_df["Close"], linewidth=0.8, color="#b41f1f", label="VOO")

# Set title and axis labels
ax.set_title("PG versus VOO Stock Price History", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Normalized Closing Price\n(Base = 100, Starting from VOO IPO Date)", fontsize=12)

# Set x-axis date format with major ticks every year
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.YearLocator(2))
plt.xticks(rotation=45)

# Add grid lines for readability
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
ax.legend(fontsize=11)

# Display date range in the bottom-right corner
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

# Save the chart as PNG
fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
print(f"Chart saved to: {PNG_PATH}")

plt.close(fig)
print("Analysis complete.")