# Data Preparation

annotations: each line contains the boundary of the clip (start and end time) corresponding to each 0-indexed label

## Data Cleaning Steps

1. Standardize the labels by making sure all videos are annotated with the same set of labels
2. Clip the video segments after the end time of the last label