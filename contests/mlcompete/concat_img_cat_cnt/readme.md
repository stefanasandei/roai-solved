# Number of categories in concatenated images

Link: https://platform.olimpiada-ai.ro/problems/36

Random idea: 75 might be the max score??

## approach 0: 46p

- compute score as `width / 32`

## approach 1: 70p

- resize images as squares (downside: lose information)
- do a CNN with regression
