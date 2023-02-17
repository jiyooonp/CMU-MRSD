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

    p = np.zeros(6)

    It_width, It_height = It.shape
    width, height = It_width, It_height
    wh = width*height
    # It1_width, It1_height = It1.shape
    It_interp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    x = np.linspace(0, It_width, It_width)  # 59 60 61 ... 144 145
    y = np.linspace(0, It_height, It_height)  # 116 117 ... 150 151
    x_stack = np.dstack([x] * y.shape[0])
    y_stack = np.dstack([y] * x.shape[0]).T
    # grid_y, grid_x  = np.meshgrid(x, y)
    # grid_x , grid_y = np.meshgrid(x, y)
    # It_patch = It_interp.ev(grid_x, grid_y)
    It_patch = It_interp.ev(y_stack[:, :, 0], x_stack[0])

    itered = 0

    # print(f"Img info: w: {It_width} h: {It_height} wh: {It_height*It_width}")
    # print(f"Img info: w: {It1_width} h: {It1_height} wh: {It1_height*It1_width}")
    while True:
        itered += 1

        It1_w = affine_transform(It1, M)
        # print("It1_w.shape", It1_w.shape)
        # plt.imshow(It1_w_interp_patch, cmap='gray')
        # plt.show()

        It1_w_interp = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1_w)

        it1_w_x = np.linspace(0, It.shape[0], It.shape[0]) # 480
        it1_w_y = np.linspace(0, It.shape[1], It.shape[1]) # 640

        # print(it1_w_x.shape)
        # print(it1_w_y.shape)

        x1_stack = np.dstack([it1_w_x] * it1_w_y.shape[0])
        y1_stack = np.dstack([it1_w_y] * it1_w_x.shape[0]).T
        It1_w_interp_patch = It1_w_interp.ev(y1_stack[:, :, 0], x1_stack[0]) # 480 x 640
        # print(f"It1_w_interp_patch.shape: {It1_w_interp_patch.shape}")

        d_It1_w_x = It1_w_interp.ev(y1_stack[:, :, 0], x1_stack[0], dy=1).flatten() #307200
        d_It1_w_y = It1_w_interp.ev(y1_stack[:, :, 0], x1_stack[0], dx=1).flatten() #307200

        It1_w_x = It1_w_interp.ev(y1_stack[:, :, 0], x1_stack[0]).flatten()  # 307200
        It1_w_y = It1_w_interp.ev(y1_stack[:, :, 0], x1_stack[0]).flatten()  # 307200

        # print(f"It1_w_x.shape: {It1_w_x.shape}")
        # print(f"It1_w_y.shape: {It1_w_y.shape}")

        dW = np.zeros((width*height, 2, 6))
        # print("shape:",dW[:, 0, 0].shape)
        dW[:, 0, 0] = It1_w_x
        dW[:, 0, 2] = It1_w_y
        dW[:, 0, 4] = np.ones(wh)

        dW[:, 1, 1] = It1_w_x
        dW[:, 1, 3] = It1_w_y
        dW[:, 1, 5] = np.ones(wh)
        # print(dW[0, :, :])

        dI = np.zeros((width*height, 1, 2))

        dI[:, :, 0] = d_It1_w_x.reshape(wh, 1)
        dI[:, :, 1] = d_It1_w_y.reshape(wh, 1)

        A_pre = np.einsum('ijk, imj -> imk', dW, dI)
        # print(A.shape)
        A = A_pre[:, 0, :]
        # print(A.shape)

        At = np.transpose(A)
        # print(At.shape)
        H = np.dot(At, A)
        # print(f"H: {H.shape}")

        D = It.flatten() - It1_w.flatten() # wh * 1


        dp = np.dot(np.linalg.pinv(H), np.dot(At, D)).reshape(2, 3)
        p_star = np.linalg.norm(dp, ord=2)
        M += dp

        if p_star <= threshold or itered >= num_iters:

            break
    print("M:", M.flatten())
    return M
