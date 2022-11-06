import time
from os.path import join

import numpy as np
from PIL import Image

import util
import visual_recog
import visual_words
from opts import get_opts

scale_list = [[1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, pow(8, 0.5)]]
K = [10 * i for i in range(1, 4)]
alpha = [25 * i for i in range(1, 4)]
L = [i for i in range(1, 3)]


def main():
    opts = get_opts()

    # ---- customize opts ----
    opts.filter_scales = [1, 2]
    opts.K = 100
    opts.alpha = 25
    opts.L = 2

    # ---- for saving results
    f_name = str(len(opts.filter_scales)) + '_K_' + str(opts.K) + '_a_' + str(opts.alpha) + '_l_' + str(opts.L)

    # ---- for printing out current run values
    name = str(opts.filter_scales) + ' | ' + str(opts.K) + ' | ' + str(opts.alpha) + ' | ' + str(opts.L) + '|'
    print('================= doing:', name)

    # to ge time of run
    start = time.time()

    # Q1.1
    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255

    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses, name)

    # Q1.2
    n_cpu = util.get_num_CPU()
    print(n_cpu)
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # Q1.3
    img_list = ['kitchen/sun_aasmevtpkslccptd.jpg', 'desert/sun_bjmzozstvsxisvgx.jpg',
                'laundromat/sun_abxdiskbkjlsqejk.jpg', 'highway/sun_baxahfrqnlwlqxkr.jpg']
    wordmaps = []
    imgs = []
    for i in range(len(img_list)):
        img_path = join(opts.data_dir, img_list[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32) / 255
        imgs.append(img)
        dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
        wordmaps.append(visual_words.get_visual_words(opts, img, dictionary))
    # util.visualize_wordmaps(wordmaps, imgs, f_name, '../result/')

    # Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    print('Done Q2.1- 2.4')

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    print()
    print(conf)
    print(accuracy)

    # to get time of run
    end = time.time()
    total_time = end - start
    print('\ntime it took:', total_time)
    print('Done Q2.5')

    # ---- to save results of run ----
    # np.savetxt(join('../result/', f_name + 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join('../result/', f_name + 'accuracy.txt'), [accuracy], fmt='%g')
    # f = open("../result/log.txt", "a")
    # f.write(name+' '+str(accuracy)+' |![alt text](../result/'+f_name+'.png)|'+str(total_time)+'|')
    # f.close()


if __name__ == '__main__':
    main()
