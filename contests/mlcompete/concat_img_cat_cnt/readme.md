# Number of categories in concatenated images

Link: https://platform.olimpiada-ai.ro/problems/36

Random idea: 75 might be the max score??

## approach 0: 99p


- run a pretrained resnet50 on cifar10 (since the images are 32x32 from cifar10) on each tile within an image
- count the number of unique categories per image


## approach 1: 70p

- use a resnet pretraine on imagenet
- resize images as squares (downside: lose information)
- do a CNN with regression
