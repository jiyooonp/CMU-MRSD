import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

from helper import plotMatches
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from opts import get_opts


# Q2.1.6

def rotTest(opts):
    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    # Read the image and convert to grayscale, if necessary
    I1 = cv2.imread('../data/cv_cover.jpg')
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    match_list = []
    for i in range(36):
        # Rotate Image
        I2 = ndimage.rotate(I1, 10 * i, reshape=False)

        # Compute features, descriptors and Match features
        locs1 = corner_detection(I1, sigma)
        locs2 = corner_detection(I2, sigma)

        # TODO: Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBrief(I1, locs1)
        desc2, locs2 = computeBrief(I2, locs2)

        # TODO: Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)
        # print(matches.shape)
        match_list.append(matches.shape[0])

    # Update histogram
    # match_hist = plt.hist(match_list)
    plt.bar([i for i in range(36)], match_list, color='slateblue')

    # Display histogram
    # plt.hist(match_list)
    plt.show()


if __name__ == "__main__":
    opts = get_opts()
    rotTest(opts)
