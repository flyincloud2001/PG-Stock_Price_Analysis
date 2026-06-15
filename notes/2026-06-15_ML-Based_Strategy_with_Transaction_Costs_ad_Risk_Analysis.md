# SVC-Based Trading Strategy with Transaction Costs and Risk Analysis on PG Stock
I learned this method from the book *Python for Finance: Mastering Data-Driven Finance*

## Why This Model Fails
The model assumes that the past five days' return directions are predictive of the next day's direction. However, PG's price movement is close to a random walk, meaning no such predictive relationship exists. As a result, the hyperplane learned from the training data fails to meaningfully separate +1 and -1 labeled vectors in the test data.

## Transaction Cost Consideration
This code incorporates proportional transaction costs into the strategy return, providing a more realistic performance estimate.

## Risk Analysis
First, we compute the optimal Kelly leverage by dividing the annualized mean return by the annualized variance. The resulting leverage ratio for PG is approximately 0.98.
Second, by fixing the initial capital, we simulate the cumulative return under different leverage ratios applied to the daily strategy return.
Third, we compute the drawdown at each time point as the difference between the running maximum of the equity curve and the current equity value. From this drawdown series, we identify the maximum drawdown amount, the date on which it occurred, and the longest recovery period between consecutive equity highs. We also compute the daily VaR at multiple confidence levels using historical percentiles of the leveraged return series.

