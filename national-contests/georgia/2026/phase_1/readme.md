# AI League 1

Link: https://judge.nitro-ai.org/competitions/gaia/ai-league-i

This is the classic ML Round. Task statements are provided in english in the [statements](./statements/).

## Memory Trace: 100p

Explanation coming soon!

Summary: use the probabilities predicted by the model to get its confidence. Compare them to the `y` target and filter out only the rows with high confidence.


## Planet X Model Selection: 100p

Explanation coming soon!

Summary: train a linear regression model to predict the 1000 provided predictions, using the 15 features. This is the vectorized version equivalent to training 1000 models, on the 15 features, to predict each 1 target. After we scale the coefficients, do the mean to get an array of 1000 floats. Apply KKN to this and find the 5 closest models (the ones with the coefficients closest).

## Data Reduction Optimization

todo
