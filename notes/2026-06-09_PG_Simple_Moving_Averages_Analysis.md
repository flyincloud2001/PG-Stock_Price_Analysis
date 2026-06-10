# This is the first algorithm analysis on PG stock - Simple Moving Averages. This analyzes the single PG stock only.
  I learned this method from the book: Python for Finance MASTERING DATA-DRIVEN FINANCE, and wrote the code for our stock analysis. Here I provide a brief logic behind this algorithm.

## What are we looking for?
We are looking for the best two parameters, which are named sma1 and sma2 in the code.
- sma1 represents the short time period where we take the average of the stock prices.
- sma2 represents the long time period where we take the average of the stock prices.

## How are these two parameters used in the algorithm?
  Suppose sma1 = 30days, sma2 = 100days, then for any date in the stock-price time series, we calculate the average of the stock prices in the past 30 and 100 days before this date as p1 and p2. If p1 > p2, then we take a long position. On the other hand, if p1 < p2, then we take a short position.

## How do we find the best parameters sma1 and sma2 which optimize the total return during the whole time period?
  First, we fix sma1 and sma2 as adjustable variables. Second, we calculate the total return R% cumulatively multiplied by every-day return controlled by sma1 and sma2. Finally, we run over every possible values (sma1, sma2) to find the highest R%.

## How do we apply this algorithm to predict the prices?
  We split the stock-price time series into 70% and 30%. The former period is used for training to find the best sma1 and sma2, and the latter period is where the best sma1 and sma2 are applied to.
  As a result, the best two parameters (in the sense that they give the highest excess return over this period) are found as sma1 = 36 days and sma2 = 260 days in the former period of 2010–2024.
  After applying these two parameters to the latter period, our strategy gives a total return of -9.26%, while the benchmark return is +33.03%. Apparently the parameters we found do not generalize well.
