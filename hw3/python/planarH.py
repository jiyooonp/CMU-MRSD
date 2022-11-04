import math
import random as rd

import cv2
import numpy as np


def match_maker(match, loc1, loc2):
    N = match.shape[0]
    # if N > 10:
    #     N = 10
    loc1_, loc2_ = np.zeros((N, 2)), np.zeros((N, 2))
    x1, x2 = [], []
    for i in range(N):
        # loc1_[i, :] = loc1[match[i, 0], :]
        # loc2_[i, :] = loc2[match[i, 1], :]
        m1 = loc1[match[i, 0], :]
        m2 = loc2[match[i, 1], :]
        x1.append(m1)
        x2.append(m2)
    x1 = np.vstack(x1)
    x2 = np.vstack(x2)

    return x1, x2


def computeH(x1, x2):
    # Q2.2.1
    N = x1.shape[0]
    # Compute the homography between two sets of points
    A = [[[-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0]],
          [0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1]]]
         for i in range(N)
         ]
    A = np.array(A).reshape(2 * N, -1)
    u, s, vh = np.linalg.svd(A)
    H2to1 = vh[-1, :].reshape(3, 3) / vh[-1, -1]

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    x1_x = np.mean(x1[:, 0])
    x1_y = np.mean(x1[:, 1])

    x2_x = np.mean(x2[:, 0])
    x2_y = np.mean(x2[:, 1])

    # Shift the origin of the points to the centroid
    x1_n, x2_n = np.copy(x1), np.copy(x2)
    x1_n[:, 0] = x1_n[:, 0] - x1_x
    x1_n[:, 1] = x1_n[:, 1] - x1_y

    x2_n[:, 0] = x2_n[:, 0] - x2_x
    x2_n[:, 1] = x2_n[:, 1] - x2_y

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    dist1 = [np.linalg.norm(x1_n[i, :]) for i in range(x1.shape[0])]
    dist2 = [np.linalg.norm(x2_n[i, :]) for i in range(x2.shape[0])]
    s1 = math.sqrt(2) / max(dist1)
    s2 = math.sqrt(2) / max(dist2)
    x1_n = x1_n * s1
    x2_n = x2_n * s2

    # Similarity transform 1
    T1 = np.array([[s1, 0, -x1_x * s1],
                   [0, s1, -x1_y * s1],
                   [0, 0, 1]])

    # Similarity transform 2
    T2 = np.array([[s2, 0, -x2_x * s2],
                   [0, s2, -x2_y * s2],
                   [0, 0, 1]])
    # Compute homography
    H = computeH(x1_n, x2_n)

    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H @ T2
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points

    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol  # the tolerance value for considering a point to be an inlier
    # print(locs1.shape)
    locsf1 = np.fliplr(locs1)
    locsf2 = np.fliplr(locs2)

    iters = 0
    max_right = 0

    rand_len = locs1.shape[0]

    while iters < max_iters:
        iters += 1
        # print(iters, end=" ")

        # rand_points = np.random.choice(rand_len, 4)
        rand_points = np.array(rd.sample(range(rand_len), 4))

        H2to1 = computeH_norm(locsf1[rand_points, :], locsf2[rand_points, :])
        img2_warped = np.matmul(H2to1,
                                np.concatenate((locsf2, np.ones((locsf2.shape[0], 1))), axis=1).transpose())
        img2_warped = img2_warped / img2_warped[2, :]
        img2_warped = img2_warped[:2, :].T

        right = np.zeros((rand_len, 1))
        for i in range(rand_len):
            if np.linalg.norm(img2_warped[i, :] - locsf1[i, :]) < inlier_tol:
                right[i] = 1
            else:
                right[i] = 0
        if np.sum(right) > max_right:
            bestH2to1 = H2to1
            inliers = right
            max_right = np.sum(right)
    print("::", np.sum(max_right))

    bestH2to1 = bestH2to1.reshape(3, 3)
    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    mask = np.ones(template.shape)
    # Warp mask by appropriate homography
    rows, cols, ch = img.shape
    warped_mask = cv2.warpPerspective(mask, np.linalg.inv(H2to1), (cols, rows))

    # Warp template by appropriate homography
    warped_img = cv2.warpPerspective(template, np.linalg.inv(H2to1), (cols, rows))

    # Use mask to combine the warped template and the image
    composite_img = img
    composite_img[np.where(warped_mask == 1)] = warped_img[np.where(warped_mask == 1)]
    return composite_img
