import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from LucasKanade import LucasKanade
from tqdm import tqdm
import time
# write your script here, we recommend the above libraries for making your animation
def updatefig(i, plot_idx = [0,99,199,299,399]):
    rect = lk_res[i]
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    patch.set_width(pt_bottomright[0] - pt_topleft[0])
    patch.set_height(pt_bottomright[1] - pt_topleft[1])
    patch.set_xy((pt_topleft[0],pt_topleft[1]))

    rect = lk_res_prev[i]
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    patch_prev.set_width(pt_bottomright[0] - pt_topleft[0])
    patch_prev.set_height(pt_bottomright[1] - pt_topleft[1])
    patch_prev.set_xy((pt_topleft[0],pt_topleft[1]))

    im.set_array(seq[:,:,i])
    # if i in plot_idx:
    plt.savefig("result/tracking_1_4_car_" + str(i) + ".png")
    plt.savefig("result/tracking_1_4_car_" + str(i) + ".png")
    return im,

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e2, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")

rect_0 = np.array([59, 116, 145, 151]).T
rect = np.copy(rect_0)

threshold_drift = 3
####### Run LK Template correction #######
lk_res = []
lk_res.append(rect)
It0 = seq[:,:,0]
template_idx = 0
template_rect = rect
p = np.zeros(2)

p_acc = np.zeros(2)
for i in tqdm(range(1, seq.shape[2])):
    ########## TODO Implement LK with drift correction ##########
    pt_topleft = lk_res[template_idx][:2]
    pt_bottomright = lk_res[template_idx][2:4]
    print(f"\n It: {template_idx}, It1: {i}")

    # if template_idx== i-1:
    #     store_p = np.zeros(2)
    # else:
    #     store_p = rect[:2]- lk_res[template_idx][:2]
    It = seq[:, :, template_idx]
    It1 = seq[:,:,i]

    p = LucasKanade(It, It1, lk_res[template_idx], threshold, num_iters)
    rect = np.concatenate((pt_topleft + p, pt_bottomright + p))

    p_acc += p

    pn = LucasKanade(It0, It1, template_rect, threshold, num_iters, p0=np.copy(p_acc))

    error_e = np.linalg.norm(pn-p_acc)

    # print(f"\nP: {p} || Pn-p_acc: {pn-p_acc} || error_e: {error_e}")

    # print(f"rect: {(rect[0]+rect[2])/2}, {(rect[1]+rect[3])/2}")
    if error_e <= threshold_drift:
        template_idx = i

    ########## TODO Implement LK with drift correction ##########

    lk_res.append(rect)

lk_res = np.array(lk_res)
np.save("carseqrects-wcrt.npy", lk_res)

####### Visualization #######

lk_res = np.load("carseqrects-wcrt.npy")
lk_res_prev = np.load("carseqrects.npy")

fig,ax = plt.subplots(1)
It1 = seq[:,:,0]
rect = lk_res[0]
pt_topleft = rect[:2]
pt_bottomright = rect[2:4]
patch = patches.Rectangle((pt_topleft[0],pt_topleft[1]), pt_bottomright[0] - pt_topleft[0],pt_bottomright[1] - pt_topleft[1] ,linewidth=2,edgecolor='r',facecolor='none')
rect = lk_res[0]
pt_topleft = rect[:2]
pt_bottomright = rect[2:4]
patch_prev = patches.Rectangle((pt_topleft[0],pt_topleft[1]), pt_bottomright[0] - pt_topleft[0],pt_bottomright[1] - pt_topleft[1] ,linewidth=2,edgecolor='purple',facecolor='none')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.add_patch(patch)
ax.add_patch(patch_prev)
im = ax.imshow(It1, cmap='gray')

ani = animation.FuncAnimation(fig, updatefig, frames=range(lk_res.shape[0]), 
                            interval=50, blit=True)

plt.show() 


### Sample code for genearting output image grid
fig, axarr = plt.subplots(1, 5)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plot_idx = [0,99,199,299,399]
for i in range(5):
    axarr[i].imshow(plt.imread(f"result/tracking_1_4_car_" + str(plot_idx[i]) + ".png"))
    axarr[i].axis('off'); axarr[i].axis('tight'); axarr[i].axis('image'); 
plt.show()