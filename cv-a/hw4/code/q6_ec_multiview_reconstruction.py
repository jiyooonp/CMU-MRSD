import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors, plot_3d_keypoint1
from q3_2_triangulate import triangulate, triangulate3D
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

    thresh = Thres
    err = 0

    realP = np.zeros((N, 3))
    for i in range(N):
        print("::", pts1[i, 2], pts2[i, 2], pts3[i, 2])
        if pts1[i, 2] > thresh and pts2[i, 2] > thresh and pts3[i, 2] > thresh:
            X, error = triangulate3D(C1, pts1[i, :2].reshape(1, 2), C2, pts2[i, :2].reshape(1, 2), C3,
                                     pts3[i, :2].reshape(1, 2))
        elif pts1[i, 2] > thresh and pts2[i, 2] > thresh:
            X, error = triangulate(C1, pts1[i, :2].reshape(1, 2), C2, pts2[i, :2].reshape(1, 2))
        elif pts1[i, 2] > thresh and pts3[i, 2] > thresh:
            X, error = triangulate(C1, pts1[i, :2].reshape(1, 2), C3, pts3[i, :2].reshape(1, 2))
        elif pts2[i, 2] > thresh and pts3[i, 2] > thresh:
            X, error = triangulate(C2, pts2[i, :2].reshape(1, 2), C3, pts3[i, :2].reshape(1, 2))
        else:
            print("not working!!!")
        realP[i, :] = X
        err += error
    np.savez('results/q6_1.npz', P=realP)
    return realP, err


'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''

connections_3d = [[0, 1], [1, 3], [2, 3], [2, 0], [4, 5], [6, 7], [8, 9], [9, 11], [10, 11], [10, 8], [0, 4], [4, 8],
                  [1, 5], [5, 9], [2, 6], [6, 10], [3, 7], [7, 11]]
color_links = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0), (0, 255, 0),
               (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 255),
               (255, 0, 255), (255, 0, 255), (255, 0, 255)]
colors = ['blue', 'blue', 'blue', 'blue', 'red', 'magenta', 'green', 'green', 'green', 'green', 'red', 'red', 'red',
          'red', 'magenta', 'magenta', 'magenta', 'magenta']


def plot_3d_keypoint_video(pts_3d_video):
    # Replace pass by your implementation
    fig = plt.figure()
    num_points = len(pts_3d_video)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_points):
        pts_3d = pts_3d_video[i]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])
    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


# Extra Credit
if __name__ == "__main__":

    pts_3d_video = []
    for loop in range(1):
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

        # print("pts1")
        # print(pts1)

        # Note - Press 'Escape' key to exit img preview and loop further
        # img1 = visualize_keypoints(im1, pts1)
        # img2 = visualize_keypoints(im2, pts2)
        # img3 = visualize_keypoints(im3, pts3)

        # YOUR CODE HERE

        C1 = K1.dot(M1)
        C2 = K2.dot(M2)
        C3 = K3.dot(M3)

        N = pts1.shape[0]

        P, err = MultiviewReconstruction(K1, pts1, K2, pts2, K3, pts3, imgs, Thres=300)
        print("error:", err)

        plot_3d_keypoint(P)
        pts_3d_video.append(P)
    # plot_3d_keypoint_video(pts_3d_video)
