import math

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

    N = pts1.shape[0]

    X = np.zeros((N, 3))
    error = 0

    Proj1_n_list = np.zeros((N, 2))
    Proj2_n_list = np.zeros((N, 2))
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
        X[i, :] = P[:3]

        pt3D = P
        Proj1 = C1 @ pt3D.T  # 3xN
        Proj1_n = (Proj1 / Proj1[2])[:2]  # 2xN
        Proj1_n_list[i, :] = Proj1_n

        Proj2 = C2 @ pt3D.T  # 3xN
        Proj2_n = (Proj2 / Proj2[2])[:2]  # 2xN
        Proj2_n_list[i, :] = Proj2_n

        e1 = (np.linalg.norm(Proj1_n - pts1[i, :])) ** 2
        e2 = (np.linalg.norm(Proj2_n - pts2[i, :])) ** 2
        # print(e1, e2)
        error += (e1 + e2)

    # err = np.sum((pts1[:, 0] - Proj1_n_list[:, 0]) ** 2 + (pts1[:, 1] - Proj1_n_list[:, 1]) ** 2 + (
    #         pts2[:, 0] - Proj2_n_list[:, 0]) ** 2 + (pts2[:, 1] - Proj2_n_list[:, 1]) ** 2)
    # print(error, err)
    return X, error


def triangulate3D(C1, pts1, C2, pts2, C3, pts3):
    # (1)
    N = pts1.shape[0]

    X = np.zeros((N, 3))
    error = 0

    Proj1_n_list = np.zeros((N, 2))
    Proj2_n_list = np.zeros((N, 2))
    Proj3_n_list = np.zeros((N, 2))

    for i in range(N):
        A = np.array([
            pts1[i, 1] * C1[2, :] - C1[1, :],
            C1[0, :] - pts1[i, 0] * C1[2, :],
            pts2[i, 1] * C2[2, :] - C2[1, :],
            C2[0, :] - pts2[i, 0] * C2[2, :],
            pts3[i, 1] * C3[2, :] - C3[1, :],
            C3[0, :] - pts3[i, 0] * C3[2, :]
        ])
        u, s, vh = np.linalg.svd(A)
        P = vh[-1, :].reshape(4, -1)
        P /= P[-1, :]
        P = P.reshape(4)
        X[i, :] = P[:3]

        pt3D = P
        Proj1 = C1 @ pt3D.T  # 3xN
        Proj1_n = (Proj1 / Proj1[2])[:2]  # 2xN
        Proj1_n_list[i, :] = Proj1_n

        Proj2 = C2 @ pt3D.T  # 3xN
        Proj2_n = (Proj2 / Proj2[2])[:2]  # 2xN
        Proj2_n_list[i, :] = Proj2_n

        Proj3 = C3 @ pt3D.T  # 3xN
        Proj3_n = (Proj3 / Proj3[2])[:2]  # 2xN
        Proj3_n_list[i, :] = Proj3_n

        e1 = (np.linalg.norm(Proj1_n - pts1[i, :])) ** 2
        e2 = (np.linalg.norm(Proj2_n - pts2[i, :])) ** 2
        e3 = (np.linalg.norm(Proj3_n - pts3[i, :])) ** 2
        error += (e1 + e2 + e3)

    return X, error


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

    M2s = camera2(E)
    min_p = -10000000
    M2 = None

    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)

    Pf = None
    C2f = None

    for i in range(M2s.shape[-1]):
        m = M2s[:, :, i]
        C2 = K2.dot(m)
        P, err = triangulate(C1, pts1, C2, pts2)

        if np.min(P[:, -2]) > min_p or np.min(P[:, -2]) > 0:
            min_p = np.min(P[:, -2])
            M2 = m
            Pf = P
            C2f = C2
    # save
    np.savez('results/' + filename, M2=M2, C2=C2f, P=Pf)

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
    print("M2:", M2)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    assert (err < 500)
