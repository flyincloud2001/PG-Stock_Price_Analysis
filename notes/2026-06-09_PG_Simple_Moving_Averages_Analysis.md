#This is the first algorithm analysis on PG stock. This analyzes the single PG stock only.
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
  We split the stock-price time series as 70% and 30%. The former period is used for training to find the best sma1 and sam2, and the latter period is where the best sma1 and sma2 are applied to.
  As a result, the best sma1 and sma2 found in the former period give a total return R% = 204% for PG stock over the 70% former period of 2010-2024, while the actual return over this period is 247% if we keep holding this stock without selling it.
  After applied to the latter period, our algorithm gives a total return as R% = 91%, while the actual return is 133%. Apparently the algorithm does not predict well on this product.
