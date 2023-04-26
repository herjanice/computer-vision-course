import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

np.random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # cv2.imshow("start", im1)
        # cv2.waitKey(0)

        # cv2.imshow("start", im2)
        # cv2.waitKey(0)

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        im1_keypoints, im1_desc = orb.detectAndCompute(im1, None)
        im2_keypoints, im2_desc = orb.detectAndCompute(im2, None)

        matcher = cv2.BFMatcher()
        matches = matcher.match(im1_desc, im2_desc)

        points1 = np.array([im1_keypoints[m.queryIdx].pt for m in matches])
        points2 = np.array([im2_keypoints[m.trainIdx].pt for m in matches])

        real_H,_ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        # TODO: 2. apply RANSAC to choose best H
        best_H = None
        best_error = 10000
        best_inliers = 0

        for trial in range(10000):
            # rand = np.random.default_rng()
            # randomSample = rand.choice(len(points1), size=4, replace=False)
            randomSample = [np.random.randint(0,len(points2)) for i in range(4)]
            
            chosen_points1 = points1[randomSample]
            chosen_points2 = points2[randomSample]

            H = solve_homography(chosen_points2, chosen_points1)
            
            ones = np.ones(points2.shape[0])
            coordinates = np.stack((points2[:,0], points2[:,1], ones))

            new_coordinates = np.matmul(H, coordinates)

            x = np.array(new_coordinates[0] / new_coordinates[2]).astype(int)
            y = np.array(new_coordinates[1] / new_coordinates[2]).astype(int)

            prediction = np.stack((x,y), axis=1)

            error = ((points1[:,0]-x)**2 + (points1[:,1]-y)**2)**0.5

            inliers = np.where(error < 5, 1, 0)
            num_inliers = np.sum(inliers)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_H = H

        # TODO: 3. chain the homographies
        last_best_H = np.matmul(last_best_H, best_H)

        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
        # cv2.imshow("final", dst)
        # cv2.waitKey(0)

    out = dst
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)