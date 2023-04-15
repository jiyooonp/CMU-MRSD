from sklearn.manifold import TSNE
import torch
import utils
import numpy as np
from train_q2 import ResNet
from voc_dataset import VOCDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import random

plt.figure(figsize=(15, 10))

resnet = ResNet(len(VOCDataset.CLASS_NAMES))
resnet.load_state_dict(torch.load('ccheckpoint-model-epoch50.pth'))
resnet.eval()

test_loader = utils.get_data_loader('voc', train=True, batch_size=1, split='test', inp_size=224)
count = 0

features = []
targets = []
with torch.no_grad():
    for num, (data, target, _) in enumerate(test_loader):
        feat = resnet(data)
        features.append(feat.flatten())
        targets.append(target.flatten())
        count += 1
        if count == 1000:
            print("done 1000 images")
            break

features = np.vstack(features)
targets = np.vstack(targets)
tsne = TSNE(n_components=2, verbose=1)
feats = tsne.fit_transform(features)

number_of_colors = 20
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(number_of_colors)]
colors = random.sample(colors, k=number_of_colors)

gt_colors = list()

for i in range(len(targets)):
    color_of_this = [color for ind, color in enumerate(colors) if targets[i, ind]==1]
    color_stack = np.vstack([np.array(mcolors.to_rgb(c)) for c in color_of_this])
    average_color = np.mean(color_stack, axis=0)
    gt_colors.append(average_color)
gt_colors = np.vstack(gt_colors)

test_loader = utils.get_data_loader('voc', train=True, batch_size=1, split='test', inp_size=224)
test_dataset = test_loader.dataset
class_names = test_dataset.CLASS_NAMES

legend = [
    Line2D([0], [0], marker='o', color=colors[i], markerfacecolor=colors[i], label=class_names[i], markersize=15) for
        i in range(len(class_names))]

plt.scatter(x=feats[:, 0], y=feats[:, 1], s=10, c=gt_colors)
plt.legend(handles=legend)
plt.show()
