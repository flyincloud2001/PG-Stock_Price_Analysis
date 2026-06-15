# SVC-Based Trading Strategy with Transaction Costs and Risk Analysis on PG Stock
I learned this method from the book *Python for Finance: Mastering Data-Driven Finance*

## Why This Model Fails
The model assumes that the past five days' return directions are predictive of the next day's direction. However, PG's price movement is close to a random walk, meaning no such predictive relationship exists. As a result, the hyperplane learned from the training data fails to meaningfully separate +1 and -1 labeled vectors in the test data.

## Transaction Cost Consideration
This code incorporates proportional transaction costs into the strategy return, providing a more realistic performance estimate.