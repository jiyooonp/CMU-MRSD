import cv2
import numpy as np
from PIL import Image

from displayMatch import plotMatches
from helper import loadVid
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac, compositeH, match_maker

# Import necessary functions

opts = get_opts()
opts.ratio = 0.7  # 'ratio for BRIEF feature descriptor'
opts.sigma = 0.05  # 'threshold for corner detection using FAST feature detector'

path1 = "../data/ar_source.mov"
path2 = "../data/book.mov"

ar_source = loadVid(path1)
book = loadVid(path2)

book_cover = cv2.imread('../data/cv_cover.jpg')

ratio = book_cover.shape[0] / book_cover.shape[1]
crop_y = ar_source.shape[1] - 100
crop_x = int(crop_y / ratio)

cropped = np.array(
    [a[50:crop_y, ar_source.shape[0] // 2 - crop_x // 2:ar_source.shape[0] // 2 + crop_x // 2, :] for a in
     ar_source])
cropped_scaled = np.array(
    [cv2.resize(c, (book_cover.shape[1], book_cover.shape[0])) for c in
     cropped])

N = min(ar_source.shape[0], book.shape[0])

cap = cropped_scaled
count = 0
missing = [79, 395, 435, 436, 437, 438]
composite_img_list = []
while count < N:
    if count in missing:
        print(count)

        frame = cap[count]

        matches, locs1, locs2 = matchPics(book_cover, book[count], opts)

        plotMatches(book_cover, book[count], matches, locs1, locs2)
        loc1_, loc2_ = match_maker(matches, locs1, locs2)
        # if len(loc1_) < 4:
        #     count = count + 1
        #     continue
        bestH2to1, inliers = computeH_ransac(loc1_, loc2_, opts)

        composite_img = compositeH(bestH2to1, cropped_scaled[count], book[count])
        print(composite_img.shape, cropped_scaled[count].shape, book[count].shape)
        composite_img = composite_img[:, :, [2, 1, 0]]
        cv2.imshow('plz work', composite_img)
        cv2.imwrite("frame%d.jpg" % count, composite_img)
        composite_img_list.append(composite_img)

        im = Image.fromarray(composite_img)
        im.save("../result/vid5/" + str(count) + ".jpg")
        count = count + 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        count = count + 1
        continue

cv2.destroyAllWindows()  # destroy all opened windows
