import multiprocessing
import os
from os.path import join

import numpy as np
import scipy.ndimage
import skimage.color
import sklearn.cluster
from PIL import Image


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    filter_scales = opts.filter_scales

    # ----- TODO -----
    # check if img values are within 0-1
    if img.max() > 1:
        img = np.array(img).astype(np.float32) / 255

    # check if it has three channels
    if img.shape[-1] == 1 or len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    if img.shape[-1] >= 3:
        img = img[:, :, :3]

    # ---- CHANGE ----
    # img = img[::3, ::3, :]

    # convert to lab_img
    lab_img = skimage.color.rgb2lab(img)

    filter_bank = 4 * len(filter_scales)
    filter_responses = np.zeros((img.shape[0], img.shape[1], filter_bank * 3))

    for scale in range(len(filter_scales)):
        for i in range(3):
            filter_responses1 = scipy.ndimage.gaussian_filter(lab_img[:, :, i], filter_scales[scale])
            filter_responses2 = scipy.ndimage.gaussian_laplace(lab_img[:, :, i], filter_scales[scale])
            filter_responses3 = scipy.ndimage.gaussian_filter(lab_img[:, :, i], filter_scales[scale], [0, 1])
            filter_responses4 = scipy.ndimage.gaussian_filter(lab_img[:, :, i], filter_scales[scale], [1, 0])

            filter_responses[:, :, 12 * scale + i] = filter_responses1
            filter_responses[:, :, 12 * scale + i + 3] = filter_responses2
            filter_responses[:, :, 12 * scale + i + 6] = filter_responses3
            filter_responses[:, :, 12 * scale + i + 9] = filter_responses4

    return filter_responses


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """

    # ----- TODO -----
    opts, data_dir, filename, alpha = args

    img = Image.open(join(data_dir, filename))
    img = np.array(img).astype(np.float32) / 255

    # ---- CHANGE ----
    # img = img[::3, ::3, :]

    filter_responses = extract_filter_responses(opts, img)
    x = np.random.choice(filter_responses.shape[0], alpha)
    y = np.random.choice(filter_responses.shape[1], alpha)
    cropped = np.array(filter_responses[x, y, :])

    np.savez("../temp/" + filename.split('/')[1].split('.')[0] + '.npz', a=cropped)


def compute_dictionary(opts, n_worker=4):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()

    # ----- TODO -----
    n_train_files = len(train_files)

    # make room for storing temp files
    os.makedirs("../temp/", exist_ok=True)

    args = zip([opts] * n_train_files, [data_dir] * n_train_files, train_files, [alpha] * n_train_files)

    # multi-processing
    p = multiprocessing.Pool(n_worker * 3)
    p.map(compute_dictionary_one_image, args)

    # load saved data
    filter_responses = []
    for filename in train_files:
        saved = np.load(join("../temp/", filename.split('/')[1].split('.')[0] + '.npz'))
        # print('saved shape:',saved['a'].shape)
        filter_responses.append(saved['a'])
    filter_responses = np.concatenate(filter_responses)  # changed from concatenate
    # print('filter_response shape:', filter_responses.shape)

    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    # print(dictionary)

    # check for shape
    # print('dictionary shape:',dictionary.shape)

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    return dictionary


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----
    filtered_img = extract_filter_responses(opts, img)
    h, w, d = filtered_img.shape
    picture = np.zeros([h, w])

    # ---- CHANGE ----
    filter_response = filtered_img.reshape(filtered_img.shape[0]*filtered_img.shape[1], -1)
    d = scipy.spatial.distance.cdist(filter_response, dictionary)
    wordmap = np.argmin(d, axis=1).reshape(filtered_img.shape[0], filtered_img.shape[1])

    # take image by pixel -> assign dictionary index to each value
    # for i in range(h):
    #     for j in range(w):
    #         pixel_img = filtered_img[i, j, :]
    #         # print(dictionary.shape, np.array([pixel_img]).shape)
    #         dist = scipy.spatial.distance.cdist(dictionary, np.array([pixel_img]),
    #                                             'euclidean')  # change added euclidian
    #         picture[i, j] = np.argmin(dist, axis=0)
    # wordmap = np.array(picture)
    return wordmap
