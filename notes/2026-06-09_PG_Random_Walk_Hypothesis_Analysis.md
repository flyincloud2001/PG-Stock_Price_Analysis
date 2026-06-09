# This is the second algorithm analysis on the PG stock - Random Walk Hypothesis
  I learned this method from the book: Python for Finance MASTERING DATA-DRIVEN FINANCE, and wrote the code for our stock analysis. Here I provide a brief logic behind this algorithm.

## Is this algorithm used for prediction?
  No, we are trying to verify a hypothesis instead of making a prediction.

## What is the hypothesis?
  The Random Walk Hypothesis states that stock prices are only highly correlated between today and yesterday, not any days before. Therefore, if we write y = β_0 + β_1* day1 + β_2* day2 + β_3* day3 + β_4* day4 + β_5* day5, where day1 means yesterday, and day2 means the day before yesterday etc, and β_0 is a constant intercept term that allows the model to fit the data without being forced through the origin.

## When is the hypothesis valid?
  The hypothesis is valid if β_1 is close to 1, and β_2 to β_5 are close to 0. This simply means only the yesterday term matters!

## Our result
  For PG stock, the hypothesis is verified to be valid as we get β_1 = 0.924656, and the average of β_2 to β_5 is 0.042393. This concludes that the best prediction on today's price is based on yesterday's price.