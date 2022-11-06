import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine, LucasKanadeAffine1, LucasKanadeAffine2
from InverseCompositionAffine import InverseCompositionAffine
import matplotlib.pyplot  as plt

from scipy.interpolate import RectBivariateSpline

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################

    M = LucasKanadeAffine1(image1, image2, threshold, num_iters) # gives M that makes img2 to match img1
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters) # gives M that makes img2 to match img1
    img2_m = affine_transform(image2, M)

    # diff = (img2_m - image1).flatten()
    diff = img2_m - image1
    diff = abs(diff)
    diff = diff.flatten()


    # w, h = image1.shape
    # wh = w*h
    #
    # x = np.linspace(0, w, w)
    # y = np.linspace(0, h, h)
    # grid_x, grid_y = np.meshgrid(x, y)
    #
    # new_grid_x = M[0, 0] * grid_x.flatten() + M[0, 1] * grid_y.flatten() + M[0, 2]
    # new_grid_y = M[1, 0] * grid_x.flatten() + M[1, 1] * grid_y.flatten() + M[1, 2]
    # # print(new_grid_y)
    # new_grid_xy = np.zeros((wh, 5))
    #
    # new_grid_xy[:, 0] = new_grid_x
    # new_grid_xy[:, 1] = new_grid_y
    # new_grid_xy[:, 2] = grid_x.flatten()
    # new_grid_xy[:, 3] = grid_y.flatten()
    # new_grid_xy[:, 4] = diff
    #
    # invalid_x_grid_ind = np.concatenate(
    #     (np.argwhere(0 >= new_grid_x).flatten(), np.argwhere(new_grid_x >= w).flatten()), 0)
    # invalid_y_grid_ind = np.concatenate(
    #     (np.argwhere(0 >= new_grid_y).flatten(), np.argwhere(new_grid_y >= h).flatten()), 0)
    #
    # invalid_xy_ind = np.unique(np.concatenate((invalid_x_grid_ind, invalid_y_grid_ind), 0))
    # invalid_xy_ind = np.sort(invalid_xy_ind)
    # deleted = 0
    #
    # for i in invalid_xy_ind:
    #     deleted += 1
    #     np.delete(new_grid_xy[:, 0], i - deleted)
    #     np.delete(new_grid_xy[:, 1], i - deleted)
    #     np.delete(new_grid_xy[:, 2], i - deleted)
    #     np.delete(new_grid_xy[:, 3], i - deleted)
    #     diff[i] = 0

    diff = diff.reshape(w,h)
    mask[diff > tolerance] = 1
    mask[diff < tolerance] = 0
    mask = mask.flatten()
    mask[invalid_xy_ind] = 0
    mask = mask.reshape(w, h)


    mask = binary_erosion(mask)
    # struct2 = generate_binary_structure(2, 2)
    mask = binary_dilation(mask, iterations=3)

    plt.subplot(3, 2, 1)
    plt.imshow(image1)
    plt.title("image 1")

    plt.subplot(3, 2, 2)
    plt.imshow(image2)
    plt.title("image 2")

    plt.subplot(3, 2, 3)
    plt.imshow(img2_m)
    plt.title("img1_m")

    plt.subplot(3, 2, 4)
    plt.imshow(diff)
    plt.title("diff")

    plt.subplot(3, 2, 5)
    plt.imshow(mask)
    plt.title("mask")
    plt.show()

    return mask.astype(bool)
