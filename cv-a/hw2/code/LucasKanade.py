import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
# import cv2
from matplotlib import pyplot as plt

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    # threshold = 0.05

    # plt.imshow(It, cmap='gray')
    # plt.show()

    x1, y1, x2, y2 = rect
    rect_size = [int(x2-x1), int(y2-y1)]
    p = p0

    It_interp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]),It)
    It1_interp = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]),It1)

    x = np.linspace(x1,x2, rect_size[0]) # 59 60 61 ... 144 145
    y = np.linspace(y1,y2, rect_size[1]) # 116 117 ... 150 151
    # x_stack = np.dstack([x]*y.shape[0])
    # y_stack = np.dstack([y]*x.shape[0]).T
    # grid_y, grid_x  = np.meshgrid(x, y)
    grid_x , grid_y = np.meshgrid(x, y)
    It_patch = It_interp.ev(grid_y, grid_x)
    # It_patch = It_interp.ev(y_stack[:,:,0],x_stack[0])

    itered = 0

    while True:
        itered += 1
        x_it1 = np.linspace(x1 + p[0], x2 + p[0], rect_size[0])
        y_it1 = np.linspace(y1 + p[1], y2 + p[1], rect_size[1])
        grid_x_it1, grid_y_it1 = np.meshgrid(x_it1, y_it1)
        It1_interp_patch = It1_interp.ev(grid_y_it1, grid_x_it1)
        # It1_interp_patch = It1_interp.ev(grid_x_it1, grid_y_it1)
        # x1_stack = np.dstack([x_it1] * y_it1.shape[0])
        # y1_stack = np.dstack([y_it1] * x_it1.shape[0]).T
        # It1_interp_patch = It1_interp.ev(y1_stack[:,:,0],x1_stack[0])

        Ix = It1_interp.ev(grid_y_it1, grid_x_it1 , dy=1).flatten()
        Iy = It1_interp.ev(grid_y_it1, grid_x_it1 , dx=1).flatten()
        # Ix = It1_interp.ev(y1_stack[:,:,0],x1_stack[0] , dy=1).flatten()
        # Iy = It1_interp.ev(y1_stack[:,:,0],x1_stack[0] , dx=1).flatten()
        # Ix = ndimage.sobel(It1_interp_patch, axis=0, mode='constant').flatten()
        # Iy = ndimage.sobel(It1_interp_patch, axis=1, mode='constant').flatten()

        # Ix, Iy = np.gradient(It1_interp_patch)
        # print('Ix.shape, Iy.shape: ',Ix.shape, Iy.shape)

        A = np.zeros((rect_size[0]*rect_size[1], 2))
        A[:, 0] = Ix.flatten()
        A[:, 1] = Iy.flatten()

        B = np.zeros((rect_size[0]*rect_size[1], 1))

        It_d =  It_patch.flatten()-It1_interp_patch.flatten()
        B = It_d

        At = np.transpose(A)
        [u,v] = np.dot(np.linalg.pinv(np.dot(At, A)),np.dot( At, B))
        p_star = np.linalg.norm([u,v], ord=2)
        p[0] += u
        p[1] += v
        # print(u, v, p)
        if p_star<= threshold or itered>=num_iters:
            # print('\n#, u, v:',itered,  u, v)
            # plt.imshow(It1_interp_patch, cmap='gray')
            # plt.show()
            break

    # print('p:', p)
    return p
# img_array = np.load("../data/carseq.npy")
# from matplotlib import pyplot as plt
#
# plt.imshow(img_array[:, :, 0], cmap='gray')
# plt.show()