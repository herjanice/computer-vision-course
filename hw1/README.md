# Homework 1:
results and finding of this homework could be found [here](https://github.com/herjanice/computer-vision-course/blob/main/hw1/report.pdf)

## Part 1: Scale Invariant Feature Detection
Task: Implement Difference of Gaussian

**Step 1:**
Filter images with different sigma values (5 images per octave, 2 octaves in total). In the second octave, down sample the fifth blurred image in the first octave as the base image.
**Step 2:**
Subtract the less blurred image to the first image to get the DoG
**Step 3:**
Threshold the pixel value and find the local extremum
**Step 4:**
Delete duplicate keypoints

## Part 2: Image Filtering
Task:  Implement a bilateral filter and advanced color-to-gray conversion

The joint bilateral filter is implemented in the [JBF.py](https://github.com/herjanice/computer-vision-course/blob/main/hw1/JBF.py)

