# This is the third algorithm analysis on the PG stock - Clustering and Frequency Approach
  I learned this method from the book *Python for Finance: Mastering Data-Driven Finance*, and wrote the code for our stock analysis. Here I provide a brief explanation of the logic behind these two algorithms - K-Means Clustering and the Frequency Approach.

## K-Means Clustering
  First, we shift the daily returns backward by one day and two days to create lag1-return and lag2-return, giving us a time series of return pairs. We then plot these pairs as points on a 2-dimensional plane. Second, we repeat the following process 10 times - randomly place a pair of cluster centers on the plane, group all points into two clusters based on which cluster center they are closer to, then compute the sum of squared differences between each point and its assigned cluster center. We take the result with the minimum total sum of squared differences among these 10 attempts. At this point, every point on the plane has been assigned to a unique cluster. This assignment becomes our strategy for labeling each day's price movement as going up or going down.

## Frequency Approach
  For this approach, we again create two time series, lag1-return and lag2-return, and convert each return into +1 or -1 based on sign(log(return)). We then take the sum of these two converted time series. Our strategy is, for dates where the sum equals 2, we predict the price will go down, and for all other cases, we predict it will go up. This means we expect (though not necessarily correctly) the price to go up if it went up on both of the previous two days, since this case has shown a slightly higher frequency of downward movement, and go down otherwise.

## Comments on the two strategies above
  For both strategies above, we found the hit rate is around 50% compared to actual daily returns.

## How do we predict the prices
  We implemented these two algorithms purely out of curiosity. They are simple and clearly not meaningful for accurately predicting stock prices. However, it is interesting to observe the goodness of fit when comparing the actual price time series with our strategy's cumulative return time series on in-sample data. In the future, if we devise an algorithm we believe can genuinely predict stock prices, we can apply it to in-sample data first and then test it on out-of-sample data to evaluate its performance.
