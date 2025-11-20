# Count of emojis in an image

TLDR: an image with white background has several images of random size & position pasted across it, there are also random colored shapes and lines on the image. Count the number of emojis

Solutin (100p):

- this is only an image processing task, solvable with opencv
- apply a blur and binary threshold, then count the number of connected components

For similar tasks, solve Hotspot from RoAI 2025 and "Notatie Bizantina" from ONIA 2025.
