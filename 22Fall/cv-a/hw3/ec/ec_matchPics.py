import math

import matplotlib.pyplot as plt
import skimage.color

from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from ec_briefRotTest import rotTest
import numpy as np
from scipy import ndimage


# Q2.1.4

def matchPics(I1, I2, opts):
    w, h, c = I1.shape

    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    # TODO: Convert Images to GrayScale
    I1 = skimage.color.rgb2gray(I1)
    I2 = skimage.color.rgb2gray(I2)

    # TODO: Detect Features in Both Images
    locs1 = corner_detection(I1, sigma)
    locs2 = corner_detection(I2, sigma)

    # EXTRA CREDIT try 2 (works!!!!!)
    mean_1 = np.mean(locs1, axis=0).reshape(2, 1)
    mean_2 = np.mean(locs2, axis=0).reshape(2, 1)

    or_1 = math.atan2(mean_1[0] - w / 2, mean_1[1] - h / 2)
    or_2 = math.atan2(mean_2[0] - w / 2, mean_2[1] - h / 2)

    # rotate I1 to match I2 rotation
    rad = or_1 - or_2
    deg = rad * 180 / math.pi
    I1_r = ndimage.rotate(I1, deg, reshape=True)

    locs1_r = corner_detection(I1_r, sigma)
    # TODO: Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1_r, locs1_r)
    desc2, locs2 = computeBrief(I2, locs2)

    # TODO: Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2, deg
