import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from helper import _epipoles

from q2_1_eightpoint import eightpoint


# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print('Zero line vector in displayEpipolar')

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    x1, y1 = int(x1), int(y1)

    # Replace pass by your implementation
    A = list(F @ np.array([x1, y1, 1]).T)

    [a, b, c] = [float(i) for i in A]

    w_size = 10
    search = 20

    xs = np.array([i for i in range(w_size * 2, im2.shape[0] - w_size * 2)])
    ys = -b / a * xs - c / a

    xs = [int(i) for i in xs]
    ys = [int(i) for i in ys]
    zs = zip(xs, ys)

    fzs = []
    for (zx, zy) in zs:
        if y1 - search < zx < y1 + search:
            fzs.append([zx, zy])

    img_g1 = ndimage.gaussian_filter(im1, sigma=2)
    img_g2 = ndimage.gaussian_filter(im2, sigma=2)

    d1 = img_g1[y1 - w_size: y1 + w_size, x1 - w_size: x1 + w_size]

    min_d = 1000
    [x2, y2] = [0, 0]

    for (y, x) in fzs:
        if not possible(img_g2, w_size, x, y):
            continue
        d2 = img_g2[y - w_size: y + w_size, x - w_size: x + w_size]
        d = np.sqrt(np.sum((d1 - d2) ** 2))
        if d < min_d:
            min_d = d
            [x2, y2] = [x, y]

    return x2, y2


def possible(img, w_size, x, y):
    xx = img.shape[1]
    yy = img.shape[0]

    if not (0 <= x - w_size <= xx) or not (0 <= x + w_size <= xx):
        return False
    if not (0 <= y - w_size <= yy) or not (0 <= y + w_size <= yy):
        return False
    else:
        return True


if __name__ == "__main__":
    correspondence = np.load('data/some_corresp.npz')  # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz')  # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # ----- TODO -----
    # YOUR CODE HERE
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    # assert (np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)

    # save
    filename = "q4_1.npz"
    np.savez('results/' + filename, F=F, pts1=pts1, pts2=pts2)
