import multiprocessing
import os
from copy import copy
from os.path import join

import numpy as np
from PIL import Image

import util
import visual_words


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K

    # ----- TODO -----
    hist, bins = np.histogram(wordmap.flatten(), bins=[i for i in range(K + 1)])
    hist = hist / np.sum(hist)

    # plot histogram
    # plt.hist(wordmap.flatten(), bins=[i for i in range(K+1)], density=True)
    # plt.title("histogram")
    # plt.show()
    # print(hist.shape)

    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L

    # ----- TODO -----

    # divide wordmap to parts
    # get histogram for each part, store in part_histogram
    # add the part_histograms
    # normalize
    # give them weights -> add them to final_histogram

    hist_all = np.array([])

    for l in range(L + 1):
        if l <= 1:
            weight = 2 ** (-L)
        else:
            weight = 2 ** (l - L - 1)

        parts = divide_to_parts(wordmap, l)

        for part in parts:
            part_histogram = get_feature_from_wordmap(opts, part)  # array of normalized hist
            w_part_histogram = part_histogram * weight
            hist_all = np.append(hist_all, w_part_histogram)
    hist_all = normalize(hist_all)

    return hist_all


def normalize(hist):
    hist = hist / np.sum(hist)
    return hist


def divide_to_parts(wordmap, l):
    parts = []
    h, w = wordmap.shape
    size_h = h // pow(2, l)
    size_w = w // pow(2, l)
    for i in range(pow(2, l)):
        for j in range(pow(2, l)):
            part = wordmap[size_h * i:size_h * (i + 1), size_w * j:size_w * (j + 1)]
            parts.append(part)
    return parts


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K) <== ?????
    """

    # ----- TODO -----
    data_dir = opts.data_dir

    img = Image.open(join(data_dir, img_path))
    img = np.array(img).astype(np.float32) / 255

    # ---- CHANGE ----
    # img = img[::3, ::3, :]

    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feat = get_feature_from_wordmap_SPM(opts, wordmap)  # K*(4^(L+1) - 1) / 3
    np.savez("../temp/" + img_path.split('/')[1].split('.')[0] + '.npz', a=feat)

    feature = get_feature_from_wordmap(opts, wordmap)
    return feature


def build_recognition_system(opts, n_worker=4):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    feat_size = int(opts.K * (pow(4, SPM_layer_num + 1) - 1) / 3)

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    # ----- TODO -----
    os.makedirs("../temp/", exist_ok=True)

    n_train_files = len(train_files)
    features = np.zeros([n_train_files, feat_size])

    args = zip([opts] * n_train_files, train_files, [dictionary] * n_train_files)

    # multi-processing
    p = multiprocessing.Pool(n_worker * 2)
    p.starmap(get_image_feature, args)

    for i, filename in enumerate(train_files):
        saved = np.load(join("../temp/", filename.split('/')[1].split('.')[0] + '.npz'))
        features[i, :] = saved['a']
    # features = np.array(features)
    # features.reshape()
    # print(features.shape)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
                        )


def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    similarity = np.sum(np.minimum(word_hist, histograms), axis=1)
    return similarity


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """
    C = np.zeros([8, 8])
    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)

    # ----- TODO -----
    features = trained_system["features"]

    # real_label = {0: 'aquarium', 1: 'desert', 2: 'highway', 3: 'kitchen', 4: 'laundromat', 5: 'park', 6: 'waterfall',7: 'windmill'}

    # for writeup
    imgs = [[] for i in range(8)]
    wordmaps = [[] for i in range(8)]
    descs = [[] for i in range(8)]

    for i, filename in enumerate(test_files):
        img = Image.open(join(data_dir, filename))
        img = np.array(img).astype(np.float32) / 255
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        hist_all = get_feature_from_wordmap_SPM(opts, wordmap)
        similarity = similarity_to_set(hist_all, features)
        # result = np.argmax(similarity)
        # print('result:',real_label[train_labels[result]],'|| name:',filename)
        print(i, end=' ')
        ture_result = test_labels[i]
        result = train_labels[np.argmax(similarity)]
        C[ture_result, result] += 1
        if ture_result != result and len(wordmaps[ture_result]) < 3:
            wordmaps[ture_result].append(wordmap)
            imgs[ture_result].append(img)
            descs[ture_result].append(result)

    util.visualize_wordmap_img(wordmaps, imgs, descs)

    accuracy = np.trace(C) / np.sum(C)

    return C, accuracy


def compute_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass


def evaluate_recognition_System_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass
