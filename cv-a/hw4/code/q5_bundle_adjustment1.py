import math

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarMatchGUI

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c='blue')
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''


def ransacF(pts1, pts2, M, nIters=20, tol=7):
    # Replace pass by your implementation
    max_iters = nIters  # the number of iterations to run RANSAC for
    inlier_tol = tol  # the tolerance value for considering a point to be an inlier

    iters = 0
    max_right = 0

    rand_len = pts1.shape[0]
    sampled = 140

    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)

    bestF = None

    while iters < max_iters:
        iters += 1

        # rand_points = np.random.choice(rand_len, sampled)
        rand_points = np.array(rd.sample(range(rand_len), sampled))

        pts1_c = pts1[rand_points, :]
        pts2_c = pts2[rand_points, :]
        pts1_c, pts2_c = pts2, pts1

        # Farrays = sevenpoint(pts1_c, pts2_c, M)
        F = eightpoint(pts1_c, pts2_c, M)

        # for F in Farrays:
        M2, C2, P = findM2(F, pts1_c, pts2_c, intrinsics, filename='q3_3.npz')

        P_h = np.hstack((P, np.ones((sampled, 1))))
        Proj1 = C1 @ P_h.T  # 3xN
        Proj1_h = (Proj1 / Proj1[2, :])  # 3xN
        # proj1_homo = toHomogenous(Proj1[:, :2])
        # Proj1_n = (Proj1 / Proj1[2, :])[:2, :]  # 2xN

        Proj2 = C2 @ P_h.T  # 3xN
        Proj2_h = (Proj2 / Proj2[2, :])  # 2xN
        # proj2_homo = toHomogenous(Proj2[:, :2])
        # Proj2_n = (Proj2 / Proj2[2, :])[:2, :]  # 2xN

        pts1_homo, pts2_homo = toHomogenous(pts1), toHomogenous(pts2)
        right = np.zeros((rand_len, 1))
        # res = calc_epi_error(pts1_homo, pts2_homo, F)

        res = (pts1_c[:, 0] - Proj1_h.T[:, 0]) ** 2 + (pts1_c[:, 1] - Proj1_h.T[:, 1]) ** 2 + (
                pts2_c[:, 0] - Proj2_h.T[:, 0]) ** 2 + (pts2_c[:, 1] - Proj2_h.T[:, 1]) ** 2

        # right[res < inlier_tol] = 1
        for i in range(sampled):
            err = res[i]
            if err < inlier_tol:
                right[rand_points[i]] = 1
            else:
                right[rand_points[i]] = 0
        print(iters, '===', np.sum(right))
        if np.sum(right) > max_right:
            bestF = F
            max_right = np.sum(right)

    print(iters, "::", np.sum(max_right))

    M2, C2, P = findM2(bestF, pts1, pts2, intrinsics, filename='q3_3.npz')

    P_h = np.hstack((P, np.ones((rand_len, 1))))
    Proj1 = C1 @ P_h.T  # 3xN
    Proj1_h = (Proj1 / Proj1[2, :])  # 3xN

    Proj2 = C2 @ P_h.T  # 3xN
    Proj2_h = (Proj2 / Proj2[2, :])  # 2xN

    right = np.zeros((rand_len, 1))

    res = (pts1[:, 0] - Proj1_h.T[:, 0]) ** 2 + (pts1[:, 1] - Proj1_h.T[:, 1]) ** 2 + (
            pts2[:, 0] - Proj2_h.T[:, 0]) ** 2 + (pts2[:, 1] - Proj2_h.T[:, 1]) ** 2

    right[res < inlier_tol] = 1

    inliers = right

    np.savez('results/q5_1.npz', bestF=bestF, inliers=inliers)

    return bestF, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    I = np.diag([1, 1, 1])

    if theta == 0:
        return I
    u = r / theta
    u = u.reshape(3)
    [u1, u2, u3] = u[0], u[1], u[2]
    ux = np.array(
        [[0, -u3, u2],
         [u3, 0, -u1],
         [-u2, u1, 0]]
    )
    R = math.cos(theta) * I + (1 - math.cos(theta)) * u * u.T + math.sin(theta) * ux
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T) / 2
    p = np.array([A[2, 1], A[0, 2], A[1, 0]]).T
    s = np.linalg.norm(p)
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2
    if s == 0 and c == 1:
        return np.zeros((3, 1))
    elif s == 0 and c == -1:
        t = R + np.diag([1, 1, 1])
        for i in range(3):
            if np.sum(t[:, i]) != 0:
                v = t[:, i]
                break
        u = v / np.linalg.norm(v)
        up = u * math.pi

        if (np.linalg.norm(up) == math.pi and (u[0] == u[1] and u[2] < 0)) or ((u[0] == 0 and u[1] < 0) or u[0] < 0):
            up = -up
        r = up
    theta = math.atan2(s, c)
    if math.sin(theta) != 0:
        u = p / s
        r = u * theta
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    P, r2, t2 = x[:-6], x[-6:-3], x[-3:]
    N = P.shape[0] // 3
    P = P.reshape((N, 3))
    r2 = r2.reshape((3, 1))
    t2 = t2.reshape((3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    P = np.vstack((P.T, np.ones((1, N))))
    p1_hat = K1 @ M1 @ P
    p1_hat /= p1_hat[2, :]

    p2_hat = K2 @ M2 @ P
    p2_hat /= p2_hat[2, :]

    residuals = np.concatenate([(p1 - p1_hat[:2, :].T).reshape([-1]), (p2 - p2_hat[:2, :].T).reshape([-1])])
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''


def bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, P_init):
    # Replace pass by your implementation

    obj_start = obj_end = 0

    # ----- TODO -----
    # YOUR CODE HERE
    R2_init, T2_init = M2_init[:, :3], M2_init[:, 3:]
    r2_init = invRodrigues(R2_init)
    x = np.concatenate([P_init[:, :3].reshape((-1, 1)), r2_init.reshape((-1, 1)), T2_init]).reshape((-1, 1))

    def f(x):
        r = np.sum(rodriguesResidual(K1, M1, pts1, K2, pts2, x) ** 2)
        return r

    t = scipy.optimize.minimize(f, x)
    f = t.x

    P, r2, t2 = f[:-6], f[-6:-3], f[-3:]
    P, r2, t2 = P.reshape((-1, 3)), r2.reshape((3, 1)), t2.reshape((3, 1))
    R2 = rodrigues(r2).reshape((3, 3))
    M2 = np.hstack((R2, t2))
    obj_end = t.fun

    np.savez('results/q5.npz', f=f, M2=M2, P=P, obj_start=obj_start, obj_end=obj_end)

    return M2, P, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz')  # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz')  # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']

    correspondence = np.load('data/some_corresp.npz')  # Loading correspondences
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    print(pts1.shape)
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # F, inliers = ransacF(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    # F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    # F /= F[-1, -1]
    displayEpipolarF(im1, im2, F)
    # YOUR CODE HERE
    # F = np.load('results/q5_1.npz')['bestF'].reshape((3, 3))
    # inliers = np.load('results/q5_1.npz')['inliers']
    # inliers = [int(i) for i in inliers]
    # print(inliers)
    # epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    # assert (F.shape == (3, 3))
    # assert (F[2, 2] == 1)
    # assert (F[2, 2] == 1)
    # assert (np.linalg.matrix_rank(F) == 2)

    # YOUR CODE HERE

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())
    print(mat)

    # assert (np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    # assert (np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    # YOUR CODE HERE

    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)

    M2_init, C2, P_before = findM2(F, pts1, pts2, intrinsics, filename='q3_3.npz')
    M2, P_after, obj_start, obj_end = bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, P_before)

    # M2_init, C2, P_before = findM2(F, noisy_pts1[inliers, :], noisy_pts2[inliers, :], intrinsics, filename='q3_3.npz')
    # M2, P_after, obj_start, obj_end = bundleAdjustment(K1, M1, noisy_pts1[inliers, :], K2, M2_init,
    #                                                    noisy_pts2[inliers, :], P_before)
    plot_3D_dual(P_before, P_after)
    print("fin")
