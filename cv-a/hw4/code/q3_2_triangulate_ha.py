import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    # (1)
    N = pts1.shape[0]

    Ps = []
    for i in range(N):
        A = np.array([
            pts1[i, 1] * C1[2, :] - C1[1, :],
            C1[0, :] - pts1[i, 0] * C1[2, :],
            pts2[i, 1] * C2[2, :] - C2[1, :],
            C2[0, :] - pts2[i, 0] * C2[2, :]
        ])
        u, s, vh = np.linalg.svd(A)
        P = vh[-1, :].reshape(4, -1)
        P /= P[-1, :]
        P = P.reshape(4)
        Ps.append(P)

    Ps = np.array(Ps)

    Proj1 = C1 @ Ps.T  # 3xN
    Proj1_n = (Proj1 / Proj1[2, :])[:2, :]  # 2xN

    Proj2 = C2 @ Ps.T  # 3xN
    Proj2_n = (Proj2 / Proj2[2, :])[:2, :]  # 2xN

    # err = np.sum(
    #     np.linalg.norm(pts1 - Proj1_n.T, axis=1)
    # ) + np.sum(
    #     np.linalg.norm(pts2 - Proj2_n.T, axis=1)
    # )
    err = np.sum((pts1[:, 0] - Proj1_n.T[:, 0]) ** 2 + (pts1[:, 1] - Proj1_n.T[:, 1]) ** 2 + (
            pts2[:, 0] - Proj2_n.T[:, 0]) ** 2 + (
                         pts2[:, 1] - Proj2_n.T[:, 1]) ** 2)

    return Ps, err


def triangulate2(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    # (1)
    N = pts1.shape[0]

    b = np.array([C1[0, 3] - C1[2, 3], C1[1, 3] - C1[2, 3], C2[0, 3] - C2[2, 3], C2[1, 3] - C2[2, 3]]).T
    P = []
    for i in range(N):
        A = np.array([
            [pts1[i, 0] * C1[2, 0] - C1[0, 0], pts1[i, 0] * C1[2, 1] - C1[0, 1], pts1[i, 0] * C1[2, 2] - C1[0, 2]],
            [pts1[i, 1] * C1[2, 0] - C1[1, 0], pts1[i, 1] * C1[2, 1] - C1[1, 1], pts1[i, 1] * C1[2, 2] - C1[1, 2]],
            [pts2[i, 0] * C2[2, 0] - C2[0, 0], pts2[i, 0] * C2[2, 1] - C2[0, 1], pts2[i, 0] * C2[2, 2] - C2[0, 2]],
            [pts2[i, 1] * C2[2, 0] - C2[1, 0], pts2[i, 1] * C2[2, 1] - C2[1, 1], pts2[i, 1] * C2[2, 2] - C2[1, 2]]
        ])
        xr = list(np.linalg.pinv(A.T @ A) @ A.T @ b)
        xr.append(1)
        # xr = np.array(xr)
        # xw =

        P.append(xr)
    P = np.array(P)
    # P /= P[:, -1]
    # P = P[:, :2]

    Proj1 = C1 @ P.T  # 3xN
    Proj1_n = (Proj1 / Proj1[2, :])[:2, :]  # 2xN

    Proj2 = C2 @ P.T  # 3xN
    Proj2_n = (Proj2 / Proj2[2, :])[:2, :]  # 2xN

    err = np.sum(
        np.linalg.norm(pts1 - Proj1_n.T, axis=1)
    ) + np.sum(
        np.linalg.norm(pts2 - Proj2_n.T, axis=1)
    )

    return P, err


'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename='q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    '''
    K1, K2 = intrinsics['K1'], intrinsics['K2']

    E = essentialMatrix(F, K1, K2)
    # print("F", F.shape)

    M2s = camera2(E)
    min_err = 1000000000
    M2 = None

    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)

    Pf = None
    C2f = None
    p_min = -10000000000000
    for i in range(M2s.shape[-1]):
        m = M2s[:, :, i]
        # print("m", m)
        C2 = K2.dot(m)
        P, err = triangulate(C1, pts1, C2, pts2)
        # print(i, err)
        # print("P min", np.min(P[:, -2]))
        # if err < min_err:
        if np.min(P[:, -2]) > p_min or np.min(P[:, -2]) > 0:  # ????????????????????????
            min_err = err
            M2 = m
            Pf = P
            C2f = C2
            p_min = np.min(P[:, -2])
    # print(M2)
    # print(C2f)
    return M2, C2f, Pf


if __name__ == "__main__":
    correspondence = np.load('data/some_corresp.npz')  # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz')  # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert (err < 500)


def epipolarCorrespondence(im1, im2, F, x1, y1):
    def gkern(l=5, sig=1.):
        # creates a lxl gaussian kernel with sigma 'sig'
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.sum(kernel)
        return np.array([kernel] * 3)

    W = 10  # half of the window size
    k = gkern(2 * W)  # gaussian weighting of the window
    height, width = im1.shape[0], im1.shape[1]

    l2 = F @ np.array([x1, y1, 1]).reshape(-1, 1)  # epipolar line on im2
    num = 30  # Look at 30 pixels per above/below y1, the row entry
    y2s = np.arange(y1 - num, y1 + num)
    x2s = (-l2[1] * y2s - l2[2]) / l2[0]  # Corresponding x coord on epipolar line

    inBound = lambda y, x: True if 0 <= y < height and 0 <= x < width else False
    w1 = im1[y1 - W:y1 + W, x1 - W:x1 + W]  # * k

    # Search
    minDist = 100000000
    x2, y2 = -1, -1
    for x, y in zip(x2s, y2s):
        x, y = round(x), round(y)
        if not inBound(y - W, x - W) or not inBound(y + W, x + W):
            continue
        w2 = im2[y - W:y + W, x - W:x + W]  # * k
        dist = np.sqrt(np.sum((w1 - w2) ** 2))
        if dist < minDist:
            x2, y2 = x, y
            minDist = dist

    return x2, y2
