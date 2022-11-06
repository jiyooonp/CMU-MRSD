import cv2

from helper import plotMatches
from ec_matchPics import matchPics
from opts import get_opts
from scipy import ndimage


def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """
    # i = 10
    # for i in range(4):
    image2 = ndimage.rotate(image1, 30, reshape=True)
    matches, locs1, locs2, deg = matchPics(image1, image2, opts)

    image1 = ndimage.rotate(image1, deg, reshape=True)

    # display matched features
    plotMatches(image1, image2, matches, locs1, locs2)


if __name__ == "__main__":
    opts = get_opts()
    opts.sigma = 0.1
    # opts.ratio = 0.5
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    displayMatched(opts, image1, image2)
