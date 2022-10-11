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
    p = M.flatten()
    ################### TODO Implement Lucas Kanade Affine ###################

    w, h = It.shape
    wh = w*h

    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    grid_x , grid_y = np.meshgrid(x, y)

    It_interp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    valid = np.ones((w, h))

    itered = 0

    while True:
        itered += 1

        # IW = affine_transform(It1, M)\
        # IW = cv2.warpAffine(It1, M, (h, w))
        # valid_img = cv2.warpAffine(valid, M, (h, w)) * It1

        It1_interp = RectBivariateSpline(np.arange(w), np.arange(h), It1)

        # It1_w_interp_patch = It1_w_interp.ev(grid_y, grid_x)
        # new_grid_x = (1+p[0])*grid_x.flatten() + p[2]*grid_y.flatten() + p[4]
        # new_grid_y = p[1]*grid_x.flatten() + (p[3]+1)*grid_y.flatten() + p[5]
        new_grid_x = M[0, 0]*grid_x.flatten() + M[0, 1]*grid_y.flatten() + M[0, 2]
        new_grid_y = M[1, 0]*grid_x.flatten() + M[1, 1]*grid_y.flatten() + M[1, 2]

        new_grid_xy = np.zeros((w*h, 4))

        new_grid_xy[:, 0] = new_grid_x
        new_grid_xy[:, 1] = new_grid_y
        new_grid_xy[:, 2] = grid_x.flatten()
        new_grid_xy[:, 3] = grid_y.flatten()

        # print("||||||||||||||||||M: ", M.flatten())
        # print("\n new grid x :", new_grid_x.shape)

        # new_grid_xy = [new_grid_xy[i, j, :] for (i, j) in (x, y ) if 0<=new_grid_x[i, j]<=w and 0<=new_grid_y[i, j]<=h ]
        # print("grid : ", new_grid_xy.shape)

        valid_x_grid_ind = np.where((0<=new_grid_x)&(new_grid_x<=w))
        # new_grid_x = new_grid_x[new_grid_x<=w]
        # new_grid_x = new_grid_x[valid_x_grid_ind]
        valid_y_grid_ind = np.where((0<=new_grid_y)&(new_grid_y<=h))
        # new_grid_y = new_grid_y[valid_y_grid_ind]
        # new_grid_y = new_grid_y[new_grid_y<=h]

        valid_xy_ind = np.intersect1d(valid_x_grid_ind, valid_y_grid_ind)
        new_len = valid_xy_ind.shape[0]

        # print(valid_x_grid)
        new_grid_xy = np.array([new_grid_xy[i, :] for i in valid_xy_ind])

        # print(f"new_grid_xy: {new_grid_xy.shape}")
        # print("ee", new_grid_xy.shape)

        valid_x_w_grid = new_grid_xy[:, 0]
        valid_y_w_grid = new_grid_xy[:, 1]
        valid_x_grid = new_grid_xy[:, 2]
        valid_y_grid = new_grid_xy[:, 3]

        # print("new shape:", valid_x_grid.shape, valid_y_grid.shape)

        # valid_grid_x, valid_grid_y = np.meshgrid(new_grid_x, new_grid_y)
        # print(valid_grid_x.shape, valid_grid_y.shape)

        valid_template = It_interp.ev(valid_x_grid,valid_y_grid)
        valid_warped = It1_interp.ev(valid_x_w_grid,valid_y_grid)

        D = valid_template.flatten() - valid_warped.flatten() # wh * 1

        # plt.imshow(valid_template.reshape(h, w), cmap='gray')
        # plt.title("valid_template")
        # plt.show()
        # plt.imshow(valid_warped.reshape(h, w), cmap='gray')
        # plt.title("valid_warped")
        # plt.show()

        d_It1_x = It1_interp.ev(valid_y_w_grid, valid_x_w_grid, dy=1) #307200
        d_It1_y = It1_interp.ev(valid_y_w_grid, valid_x_w_grid, dx=1) #307200

        # print("s", d_It1_x.shape)

        d_It1_x = valid_x_w_grid.reshape(new_len, 1)
        d_It1_y = valid_y_w_grid.reshape(new_len, 1)

        # print("Grad shape:", d_It1_x.shape, d_It1_y.shape)

        # d_It1_x_w = cv2.warpAffine(d_It1_x, M, (h, w)).reshape(wh, 1)
        # d_It1_y_w = cv2.warpAffine(d_It1_y, M, (h, w)).reshape(wh, 1) #307200

        It1_w_x = new_grid_x.reshape(new_len,)
        It1_w_y = new_grid_y.reshape(new_len,)

        dW = np.zeros((new_len, 2, 6))
        dW[:, 0, 0] = It1_w_x
        dW[:, 0, 2] = It1_w_y
        dW[:, 0, 4] = np.ones(new_len)
        dW[:, 1, 1] = It1_w_x
        dW[:, 1, 3] = It1_w_y
        dW[:, 1, 5] = np.ones(new_len)

        dI = np.zeros((new_len, 1, 2))
        dI[:, :, 0] = d_It1_x
        dI[:, :, 1] = d_It1_y

        A_pre = np.einsum('ijk, imj -> imk', dW, dI)

        A = A_pre[:, 0, :]

        At = np.transpose(A)
        H = np.dot(At, A)

        dp = np.dot(np.linalg.pinv(H), np.dot(At, D)).reshape(2, 3)

        p_star = np.linalg.norm(dp, ord=2)

        M += dp
        print("dp:", dp)


        if p_star <= threshold or itered >= num_iters:
            break
        # M = p.reshape(2, 3)

    print("\nM:", M.flatten())

    return M
