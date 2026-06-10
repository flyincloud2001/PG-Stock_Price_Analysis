# This is the third algorithm analysis on the PG stock - Linear OLS Regression
  I learned this method from the book: Python for Finance: Mastering Data-Driven Finance, and wrote the code for our stock analysis. Here I provide a brief logic behind this algorithm.

## Linear Model
  We apply the linear model y = β_0 + β_1 * lag_1 + β_2 * lag_2, where lag_1 and lag_2 are the log returns from yesterday and the day before yesterday respectively. First, we use the daily log return as the label y to find the parameters (β_0, β_1, β_2). This gives a strategy series of +1 or -1 depending on whether the prediction is positive or negative. Second, we use the +1/-1 direction series derived from the daily log return as the label y to find the parameters (β_0, β_1, β_2) directly. This also gives a strategy series.

## Predictions
  We use the two strategy series to calculate the prediction accuracy. This is done by comparing each strategy series against the actual direction series, which gives 1 if they match on that day and 0 if they do not. We then take the average over the whole time period. The results are 51.64% and 52.97% respectively, which supports the random walk hypothesis.