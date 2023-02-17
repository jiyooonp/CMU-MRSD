import multiprocessing
from os.path import join

import numpy as np

from matplotlib import pyplot as plt
import opts


def get_num_CPU():
    """
    Counts the number of CPUs available in the machine.
    """
    return multiprocessing.cpu_count()


def display_filter_responses(opts, response_maps, name):
    """
    Visualizes the filter response maps.

    [input]
    * response_maps: a numpy.ndarray of shape (H,W,3F)
    """

    n_scale = len(opts.filter_scales)
    plt.figure(1)
    plt.title(name)
    for i in range(n_scale * 4):
        plt.subplot(n_scale, 4, i + 1)
        resp = response_maps[:, :, i * 3: i * 3 + 3]
        resp_min = resp.min(axis=(0, 1), keepdims=True)
        resp_max = resp.max(axis=(0, 1), keepdims=True)
        resp = (resp - resp_min) / (resp_max - resp_min)
        plt.imshow(resp)
        plt.axis("off")

    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05
    )
    plt.show()


def visualize_wordmap(wordmap, original, name, f_name, out_path=None):
    rows, columns = 2, 1
    fig = plt.figure()
    plt.axis("equal")
    plt.axis("off")
    # plt.title(name)

    fig.add_subplot(rows, columns, 1)
    plt.imshow(original)
    plt.axis('off')

    fig.add_subplot(rows, columns, 2)
    plt.imshow(wordmap)
    plt.axis('off')

    out_path = out_path + f_name
    if out_path:
        plt.savefig(out_path, pad_inches=0)
    plt.show()

    plt.close()


def visualize_wordmaps(wordmaps, originals, f_name, out_path=None):
    rows, columns = 2, len(wordmaps)

    fig = plt.figure(figsize=(30, 15))
    plt.axis("equal")
    plt.axis("off")

    for i in range(1, columns + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(originals[i - 1])
        plt.axis('off')

        fig.add_subplot(rows, columns, i + columns)
        plt.imshow(wordmaps[i - 1])
        plt.axis('off')

    out_path = out_path + f_name
    if out_path:
        plt.savefig(out_path, pad_inches=0)
    plt.show()

    plt.close()


def visualize_wordmap_img(wordmaps, originals, descs):
    real_label = {0: 'aquarium', 1: 'desert', 2: 'highway', 3: 'kitchen', 4: 'laundromat', 5: 'park', 6: 'waterfall',
                  7: 'windmill'}
    for i in range(8):
        rows, columns = 2, len(wordmaps[i])
        fig = plt.figure()
        plt.axis("equal")
        plt.axis("off")

        for j in range(len(wordmaps[i])):
            name = '[' + real_label[i] + '|' + real_label[descs[i][j]] + ']'

            fig.add_subplot(rows, columns, j + 1)
            plt.imshow(originals[i][j])
            plt.axis('off')
            plt.title(name)

            fig.add_subplot(rows, columns, j + 1 + columns)
            plt.imshow(wordmaps[i][j])
            plt.axis('off')

        plt.show()

        plt.close()


def visualize_dict(opts):
    dictionary = np.load(join(opts.out_dir, "dictionary.npy"))
    print(dictionary.shape)
    print(dictionary)

    n_scale = dictionary.shape[0]
    plt.figure(1)
    for i in range(n_scale * 4):
        # print(i)
        plt(n_scale, 4, i + 1)
        resp = dictionary[:, i * 3: i * 3 + 3]
        resp_min = resp.min(axis=(0), keepdims=True)
        resp_max = resp.max(axis=(0), keepdims=True)
        resp = (resp - resp_min) / (resp_max - resp_min)
        print('resp size: ', resp.shape)

        plt.imshow(resp)
        plt.axis("off")

    # plt.subplots_adjust(
    #     left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05
    # )
    plt.show()

# opts = opts.get_opts()
# visualize_dict(opts)
