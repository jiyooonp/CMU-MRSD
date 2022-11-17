import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors, plot_3d_keypoint1
from q3_2_triangulate import triangulate
from q2_1_eightpoint import eightpoint
from q4_2_visualize import compute3D_pts
from q3_2_triangulate import findM2

# Insert your package here

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
            
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''


def MultiviewReconstruction(K1, pts1, K2, pts2, K3, pts3, imgs, Thres=100):
    # Replace pass by your implementation

    [im1, im2, im3] = imgs

    intrinsics12 = {'K1': K1, 'K2': K2}
    intrinsics13 = {'K1': K1, 'K2': K3}
    intrinsics23 = {'K1': K2, 'K2': K3}

    F12 = eightpoint(pts1[:, :2], pts2[:, :2], M=np.max([*im1.shape, *im2.shape]))
    F13 = eightpoint(pts1[:, :2], pts3[:, :2], M=np.max([*im1.shape, *im3.shape]))
    F23 = eightpoint(pts2[:, :2], pts3[:, :2], M=np.max([*im2.shape, *im3.shape]))

    M2, C2, P12 = findM2(F12, pts1[:, :2], pts2[:, :2], intrinsics12, filename='q3_3.npz')
    M2, C2, P13 = findM2(F13, pts1[:, :2], pts3[:, :2], intrinsics13, filename='q3_3.npz')
    M2, C2, P23 = findM2(F23, pts2[:, :2], pts3[:, :2], intrinsics23, filename='q3_3.npz')

    c12 = pts1[:, 2] + pts2[:, 2]
    c13 = pts1[:, 2] + pts3[:, 2]
    c23 = pts2[:, 2] + pts3[:, 2]

    realP = np.zeros(P12.shape)

    for i in range(P12.shape[0]):
        mini = max(c12[i], c13[i], c23[i])
        if c12[i] == mini:
            realP[i, :] = P12[i, :]
        elif c13[i] == mini:
            realP[i, :] = P13[i, :]
        else:
            realP[i, :] = P23[i, :]
    plot_3d_keypoint1(P12, P13, P23, realP)

    return realP


'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''


def plot_3d_keypoint_video(pts_3d_video):
    # Replace pass by your implementation
    pass


# Extra Credit
if __name__ == "__main__":

    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('data/q6/', 'time' + str(loop) + '.npz')
        image1_path = os.path.join('data/q6/', 'cam1_time' + str(loop) + '.jpg')
        image2_path = os.path.join('data/q6/', 'cam2_time' + str(loop) + '.jpg')
        image3_path = os.path.join('data/q6/', 'cam3_time' + str(loop) + '.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        imgs = [im1, im2, im3]

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        print("pts1")
        print(pts1)

        # Note - Press 'Escape' key to exit img preview and loop further
        # img1 = visualize_keypoints(im1, pts1)
        # img2 = visualize_keypoints(im2, pts2)
        # img3 = visualize_keypoints(im3, pts3)

        # YOUR CODE HERE

        C1 = K1.dot(M1)
        C2 = K2.dot(M2)
        C3 = K3.dot(M3)

        # K1, K2 = intrinsics['K1'], intrinsics['K2']

        P = MultiviewReconstruction(K1, pts1, K2, pts2, K3, pts3, imgs, Thres=100)
        # plot_3d_keypoint(P)
