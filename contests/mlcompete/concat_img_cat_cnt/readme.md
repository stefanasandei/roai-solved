# Number of categories in concatenated images

## approach 0: 46p

- compute score as `width / 32`

## approach 1: 70p

- resize images as squares (downside: lose information)
- do a CNN with regression
