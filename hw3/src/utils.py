from audioop import ulaw2lin
from tkinter import N
import numpy as np
import cv2

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []
    for i in range(N):
        u_x, u_y = u[i]
        v_x, v_y = v[i]
        
        fir = np.array([u_x, u_y, 1, 0, 0, 0, -u_x*v_x, -u_y*v_x, -v_x])
        sec = np.array([0, 0, 0, u_x, u_y, 1, -u_x*v_y, -u_y*v_y, -v_y])

        A.append(fir)
        A.append(sec)  
    A = np.array(A)

    # TODO: 2.solve H with A
    U, S, V = np.linalg.svd(A)
    H = V[-1,:].reshape(3,3)

    # Normalize
    H = (1/H.item(8)) * H

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    # cv2.imshow("source image", src)
    # cv2.waitKey(0)
    # cv2.imshow("dest image", dst)
    # cv2.waitKey(0)

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    range_x = np.arange(xmin, xmax, 1)
    range_y = np.arange(ymin, ymax, 1)
    ones = np.ones((ymax-ymin, xmax-xmin))

    x,y = np.meshgrid(range_x, range_y)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    coordinates = np.stack((x.ravel(),y.ravel(),ones.ravel())) # shape : 3 x N
    

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_pixels = np.matmul(H_inv, coordinates)
        u = np.round((new_pixels[0] / new_pixels[2]).reshape((ymax-ymin, xmax-xmin))).astype(int)
        v = np.round((new_pixels[1] / new_pixels[2]).reshape((ymax-ymin, xmax-xmin))).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = np.ones(shape=(ymax-ymin, xmax-xmin), dtype=np.bool)
        u_mask = np.where(np.logical_and(u>=0, u<w_src), True, False)
        v_mask = np.where(np.logical_and(v>=0, v<h_src), True, False)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        mask = np.logical_and(np.logical_and(u_mask, v_mask), mask)

        # TODO: 6. assign to destination image with proper masking
        dst[y[mask], x[mask]] = src[v[mask], u[mask]]

        # cv2.imshow("after backward warping", dst)
        # cv2.waitKey(0)

        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_pixels = np.matmul(H, coordinates)
        u = np.round((new_pixels[0] / new_pixels[2]).reshape((ymax-ymin, xmax-xmin))).astype(int)
        v = np.round((new_pixels[1] / new_pixels[2]).reshape((ymax-ymin, xmax-xmin))).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.ones(shape=(ymax-ymin, xmax-xmin), dtype=np.bool)
        u_mask = np.where(np.logical_and(u>=0, u<w_dst), True, False)
        v_mask = np.where(np.logical_and(v>=0, v<h_dst), True, False)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        mask = np.logical_and(np.logical_and(u_mask, v_mask), mask)

        # TODO: 6. assign to destination image using advanced array indicing
        dst[v[mask], u[mask]] = src[y[mask], x[mask]]

        # cv2.imshow('after backward warping', dst)
        # cv2.waitKey(0)

        pass
    
    return dst
