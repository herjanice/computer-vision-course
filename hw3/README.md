# Homework 3:
Results and finding of this homework could be found [here](https://github.com/herjanice/computer-vision-course/blob/main/hw3/report_B08902092.pdf)

File directory

- resource/
  - times.jpg, img1.png, img2.jpg, img3.jpg, img4.jpg, img5.png (part1)
  - img5.png, seq0.mp4 (part2)
  - BL_secret1.png,BL_secret2.png(part3)
  - frame1.jpg,frame2.jpg,frame3.jpg(part4)
- src/
  - part1.py
  - part2.py
  - part3.py
  - part4.py
  - utils.py
  - hw3.sh

## Part 1: Homography Estimation
Goal:  Gain familiarity with the Direct Linear Transformation (DLT) estimation method and practice forward warping in computer vision applications.
Task: Forward warp the 5 given image to the given template.

## Part 2: Marker-Based Planar AR
Goal: Become familiar with the off-the-shelf ArUco marker detection tool and practice backward warping in computer vision applications.
Task: Backward warp a template image onto each frame of a given video without leaving any holes, resulting in an output video that contains the warped template image.

## Part 3: Unwarp the Secret
Task: Unwarp a QR code from two different source images and retrieve the link. Discuss the differences between the two source images and compare the warped results to determine if they are the same or different. If the results are the same, explain why. If the results are different, explain why.

## Part 4: Panorama
Task: Implement the function panorama(), which estimates the homography between three images using feature matching and RANSAC to find the correct transform, then stitches the images together using backward warping. The already written function warping() should be called in part 2.
Question: Can all consecutive images be stitched into a panorama ? Why ?
