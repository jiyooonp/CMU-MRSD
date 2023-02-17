import cv2
import matplotlib.pyplot as plt
import numpy as np
from displayMatch import plotMatches
# Import necessary functions
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac, compositeH, match_maker


# Q2.2.4

def warpImage(opts):
    # Read the image and convert to grayscale, if necessary
    I1 = cv2.imread('../data/cv_cover.jpg')

    I2 = cv2.imread('../data/cv_desk.png')

    I3 = cv2.imread('../data/hp_cover.jpg')
    I3 = cv2.resize(I3, (I1.shape[1], I1.shape[0]))

    matches, locs1, locs2 = matchPics(I1, I2, opts)

    plotMatches(I1, I2, matches, locs1, locs2)
    loc1_, loc2_ = match_maker(matches, locs1, locs2)

    bestH2to1, inliers = computeH_ransac(loc1_, loc2_, opts)

    composite_img = compositeH(bestH2to1, I3, I2)
    composite_img = composite_img[:, :, [2, 1, 0]]

    plt.imshow(composite_img)
    s = str(opts.max_iters) + "," + str(opts.inlier_tol) + "," + str(np.sum(inliers))
    plt.title(s)
    plt.show()
    # plt.savefig(s + ".png")


if __name__ == "__main__":
    opts = get_opts()
    # for m in [300, 500, 1000]:
    #     for i in [20, 2, 0.5]:
    #         opts.max_iters = m
    #         opts.inlier_tol = i
    #         warpImage(opts)
    warpImage(opts)
