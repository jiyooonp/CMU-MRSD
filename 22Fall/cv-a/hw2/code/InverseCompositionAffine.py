import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    W_d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Inverse Composition Affine ###################

    w, h = It.shape
    wh = w*h

    p = np.zeros((6, 1))
    dp = np.zeros((6, 1))

    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    grid_x , grid_y = np.meshgrid(x, y)

    It_interp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    d_It_x = It_interp.ev(grid_y, grid_x, dy=1)  # 307200
    d_It_y = It_interp.ev(grid_y, grid_x, dx=1)  # 307200

    d_It_x = d_It_x.reshape(wh, 1)
    d_It_y = d_It_y.reshape(wh, 1)

    dT = np.zeros((wh, 1, 2))
    dT[:, :, 0] = d_It_x
    dT[:, :, 1] = d_It_y

    dW = np.zeros((wh, 2, 6))
    dW[:, 0, 0] = grid_x.flatten()
    dW[:, 0, 2] = grid_y.flatten()
    dW[:, 0, 4] = np.ones(wh)
    dW[:, 1, 1] = grid_x.flatten()
    dW[:, 1, 3] = grid_y.flatten()
    dW[:, 1, 5] = np.ones(wh)

    A_pre = np.einsum('ijk, imj -> imk', dW, dT)

    A = A_pre[:, 0, :]

    At = np.transpose(A)
    H = np.dot(At, A)

    A1 = np.dot(np.linalg.inv(np.dot(At, A)), At)

    itered = 0

    It1_interp = RectBivariateSpline(np.arange(w), np.arange(h), It1)

    while True:
        itered+=1
        print(itered, end=" ")

        new_grid_x = W_d[0, 0]*grid_x.flatten() + W_d[0, 1]*grid_y.flatten() + W_d[0, 2]
        new_grid_y = W_d[1, 0]*grid_x.flatten() + W_d[1, 1]*grid_y.flatten() + W_d[1, 2]

        len_new = new_grid_x.shape[0]

        new_grid_xy = np.zeros((len_new, 4))

        new_grid_xy[:, 0] = new_grid_x
        new_grid_xy[:, 1] = new_grid_y
        new_grid_xy[:, 2] = grid_x.flatten()
        new_grid_xy[:, 3] = grid_y.flatten()

        deleted = 0
        for i in range(0, len_new):
            # if 0>=new_grid_xy[i,0] or w<=new_grid_xy[i,0] or 0>=new_grid_xy[i,1] or h<=new_grid_xy[i,1]:
            if 0 <= new_grid_xy[i, 0] <=w and 0 <= new_grid_xy[i, 1] <= h:
                continue
            else:
                deleted+=1
                np.delete(new_grid_xy[:,0], i-deleted)
                np.delete(new_grid_xy[:,1], i-deleted)
                np.delete(new_grid_xy[:,2], i-deleted)
                np.delete(new_grid_xy[:,3], i-deleted)

        len_new = new_grid_x.shape[0]

        # Construct A and b matrices
        b = (It1_interp.ev(new_grid_xy[:, 3], new_grid_xy[:, 2]) - It_interp.ev(new_grid_xy[:, 1], new_grid_xy[:, 0])).reshape((len_new, 1))

        dp = np.dot(A1, b)

        dp_error = np.linalg.norm(dp)
        # print('dp_norm: ', dp_norm)

        # M = np.array([[1.0 + dp[0], dp[1], dp[2]], [dp[3], 1.0 + dp[4], dp[5]], [0.0, 0.0, 1.0]]).astype(np.float32)
        p += dp
        W_dp = np.array([[dp[0], dp[1], dp[2]], [dp[3],  dp[4], dp[5]], [0.0, 0.0, 1.0]]).astype(np.float32)
        W_p = np.array([[p[0], p[1], p[2]],
                        [p[3],  p[4], p[5]],
                        [0.0, 0.0, 1.0]]).astype(np.float32)

        W_p = np.dot(W_p, np.linalg.inv(W_dp))
        if dp_error <= threshold or itered >= num_iters:
            break

    # print("\nM:", W_p.flatten())
    print(itered)
    M0 = W_p
    return M0
