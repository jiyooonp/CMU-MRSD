import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2
from matplotlib import pyplot as plt

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Lucas Kanade Affine ###################

    w, h = It.shape
    wh = w*h

    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    grid_x, grid_y = np.meshgrid(x, y)

    It_interp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1_interp = RectBivariateSpline(np.arange(w), np.arange(h), It1)

    itered = 0

    while True:
        itered += 1

        # print("Init M:", M)
        new_grid_x = M[0, 0]*grid_x.flatten() + M[0, 1]*grid_y.flatten() + M[0, 2]
        new_grid_y = M[1, 0]*grid_x.flatten() + M[1, 1]*grid_y.flatten() + M[1, 2]

        # print('new_grid_x.shape',new_grid_x.shape)

        new_grid_xy = np.zeros((wh, 4))

        new_grid_xy[:, 0] = new_grid_x
        new_grid_xy[:, 1] = new_grid_y
        new_grid_xy[:, 2] = np.copy(grid_x).flatten()
        new_grid_xy[:, 3] = np.copy(grid_y).flatten()

        invalid_x_grid_ind = np.concatenate((np.argwhere(0>=new_grid_x).flatten(), np.argwhere(new_grid_x>=w).flatten()), 0)
        invalid_y_grid_ind = np.concatenate((np.argwhere(0>=new_grid_y).flatten(), np.argwhere(new_grid_y>=h).flatten()), 0)
        # invalid_y_grid_ind =  np.concatenate((np.argwhere(0>=new_grid_y), np.argwhere(new_grid_y>=h)), 1)
        # print(invalid_x_grid_ind)

        invalid_xy_ind = np.unique(np.concatenate((invalid_x_grid_ind, invalid_y_grid_ind), 0))
        invalid_xy_ind = np.sort(invalid_xy_ind)
        deleted = 0

        for i in invalid_xy_ind:
            deleted += 1
            np.delete(new_grid_xy[:, 0], i - deleted)
            np.delete(new_grid_xy[:, 1], i - deleted)
            np.delete(new_grid_xy[:, 2], i - deleted)
            np.delete(new_grid_xy[:, 3], i - deleted)

        # for i in range(0, len_new):
        #     # if 0>=new_grid_xy[i,0] or w<=new_grid_xy[i,0] or 0>=new_grid_xy[i,1] or h<=new_grid_xy[i,1]:
        #     if 0 <= new_grid_xy[i, 0] <=w and 0 <= new_grid_xy[i, 1] <= h:
        #         continue
        #     else:
        #         deleted+=1
        #         np.delete(new_grid_xy[:,0], i-deleted)
        #         np.delete(new_grid_xy[:,1], i-deleted)
        #         np.delete(new_grid_xy[:,2], i-deleted)
        #         np.delete(new_grid_xy[:,3], i-deleted)
        # print("deleted: ", deleted)

        # new_grid_xy = np.array([new_grid_xy[i, :] for i in valid_xy_ind])

        new_len = new_grid_xy.shape[0]

        valid_x_w_grid = new_grid_xy[:, 0]
        valid_y_w_grid = new_grid_xy[:, 1]
        valid_x_grid = new_grid_xy[:, 2]
        valid_y_grid = new_grid_xy[:, 3]

        valid_template = It_interp.ev(valid_y_grid,valid_x_grid)
        valid_warped = It1_interp.ev(valid_y_w_grid,valid_x_w_grid)

        D = valid_template.flatten() - valid_warped.flatten() # wh * 1

        d_It1_x = It1_interp.ev(valid_y_w_grid, valid_x_w_grid, dy=1) #307200
        d_It1_y = It1_interp.ev(valid_y_w_grid, valid_x_w_grid, dx=1) #307200

        d_It1_x = d_It1_x.reshape(new_len,1)
        d_It1_y = d_It1_y.reshape(new_len,1)

        # d_It1_x,  d_It1_y = np.gradient(valid_y_w_grid, valid_x_w_grid, dx=1) #307200

        valid_x_w_grid = valid_x_w_grid.reshape(new_len, )
        valid_y_w_grid = valid_y_w_grid.reshape(new_len, )

        valid_x_grid = valid_x_grid.reshape(new_len,1)
        valid_y_grid = valid_y_grid.reshape(new_len,1)

        dW = np.zeros((new_len, 2, 6))
        dW[:, 0, 0] = valid_x_w_grid
        dW[:, 0, 2] = valid_y_w_grid
        dW[:, 0, 4] = np.ones(new_len)
        dW[:, 1, 1] = valid_x_w_grid
        dW[:, 1, 3] = valid_y_w_grid
        dW[:, 1, 5] = np.ones(new_len)

        # dW[:, 0, 0] = valid_x_w_grid
        # dW[:, 0, 1] = valid_y_w_grid
        # dW[:, 0, 2] = np.ones(new_len)
        # dW[:, 1, 3] = valid_x_w_grid
        # dW[:, 1, 4] = valid_y_w_grid
        # dW[:, 1, 5] = np.ones(new_len)

        # print(valid_x_grid.shape, d_It1_x.shape)
        # A1 = d_It1_x.T* valid_x_grid.T
        # A2 = d_It1_x.T* valid_y_grid.T
        # A3 = d_It1_y.T*valid_x_grid.T
        # A4 = d_It1_y.T* valid_y_grid.T
        #
        # A = np.concatenate((A1, A2, d_It1_x.T, A3, A4, d_It1_y.T), axis=0)
        # print(A.shape)

        dI = np.zeros((new_len, 1, 2))
        dI[:, :, 0] = d_It1_x
        dI[:, :, 1] = d_It1_y

        A_pre = np.einsum('ijk, imj -> imk', dW, dI)

        A = A_pre[:, 0, :]

        At = np.transpose(A)
        H = np.dot(At, A)

        dp = np.dot(np.linalg.pinv(H), np.dot(At, D)).reshape(2, 3)

        p_error = np.linalg.norm(dp, ord=2)
        # p_error = np.sum(dp)

        M += dp
        # print("dp:", dp)

        if p_error <= threshold or itered >= num_iters:
            # print(f"P: {dp}, p_error: {p_error}")
            break

    print("\niter :", itered)
    return M
