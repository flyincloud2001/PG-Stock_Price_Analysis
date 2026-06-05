# This follows from the previous note, and we are interested in answering the following question: How and when are the VOO and PG correlated? 
Furthermore, since I recently open a position in PG stock, I am interested in another question, how do we earn from investing in this PG stock? To answer these two questions, we are motivated to build a simple software which include the following functions:
- Import the past stock prices data from Yahoo Finance
- Calculate and export the following data:
  - Total return over the selected period
  - CAGR (Compound Annual Growth Rate)
  - Correlation Coefficient between two stocks selected

## Step 1: Write the code stock_analysis_app.py
- This code utilizes the python module streamlit to create a web platform
- The web visualizes the stock price time series imported from Yahoo Finance, and calculate the data outlined at the beinging
- This project is very easily done by claude code within an hour.
- The web link is: https://pg-stockpriceanalysis-psvsaru5ab3wcrbjxhmpu9.streamlit.app/

