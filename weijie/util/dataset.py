import configparser
import os
from util import dataLoader
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('../config.ini')

    # Set Global Variables
    os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']
    print('GPU Index: [ ' + config['GLOBAL']['index_gpu'] + ' ]')

    dataset_path = config['GLOBAL']['dataset_path']
    dataset_type = config['GLOBAL']['dataset_type']

    network = config['GLOBAL']['network']
    method = config['GLOBAL']['method']

    index_train_mat = np.fromstring(config['GLOBAL']['index_train_mat'], dtype=np.int, sep=',').tolist()
    index_train_imgs = np.fromstring(config['GLOBAL']['index_train_imgs'], dtype=np.int, sep=',').tolist()

    index_valid_mat = np.fromstring(config['GLOBAL']['index_valid_mat'], dtype=np.int, sep=',').tolist()
    index_valid_imgs = np.fromstring(config['GLOBAL']['index_valid_imgs'], dtype=np.int, sep=',').tolist()

    x_train, y_train, x_val, y_val, x_train_imgs, y_train_imgs, x_val_imgs, y_val_imgs = dataLoader.trainNvalid(
        dataset_path, dataset_type, index_train_mat, index_train_imgs, index_valid_mat, index_valid_imgs)
    print('[dataLoader]: Done')

    from skimage.util import view_as_windows

    patch_size = 100
    patch_step = 10
    batchsize = x_train.shape[0]
    channel_x = x_train.shape[3]

    x_crops = view_as_windows(x_train, [batchsize, patch_size, patch_size, channel_x], patch_step)

    patch_num = x_crops.shape[1]


    def index2data(index):

        def pyfun(index_):
            # print(index_)
            return x_crops[0, index_[0], index_[1], :, index_[2], :, :, :][0]

        return tf.py_func(pyfun, [index], tf.float64, name='pyfun_index2data')


    def index_generator():
        for i in range(patch_num):
            for j in range(patch_num):
                for m in range(batchsize):
                    yield (i, j, m)
                    # yield x_crops[0, i, j, :, m, :, :, :][0]


    x_train_dataset = tf.data.Dataset.from_generator(index_generator, output_shapes=(3,), output_types=tf.float64)
    x_train_dataset = \
        x_train_dataset.shuffle(buffer_size=patch_num * patch_num * batchsize + 100).map(index2data).prefetch(100000).batch(48).repeat()

    # x_train_dataset = \
    #     x_train_dataset.shuffle(buffer_size=patch_num * patch_num * batchsize + 100).prefetch(
    #         100000).batch(48).repeat()

    x = x_train_dataset.make_one_shot_iterator().get_next()

    # print(x)
    # y = tf.layers.conv2d(x, 32, 3, padding='same')

    x_ = tf.placeholder(dtype=tf.float64, shape=[48, 100, 100, 1])
    y_ = tf.layers.conv2d(x_, 32, 3, padding='same')

    def data_g():
        return np.random.sample(size=(48, 100, 100, 1))

    import time

    init = time.time()
    # using the pyfun
    with tf.Session() as sess:
        # for i in range(1):  # 9.048
            # for i in range(1000):  # 12.118
        # for i in range(10000):  # 73.03
        sess.run(tf.global_variables_initializer())
        for i in range(1000):  # 58 with prefetch=100000 res = sess.run(y)
            res = sess.run(x)
            # data = data_g()
            # res = sess.run(y_, feed_dict={x_: data}) # only cost 23.656.... What the ....

    print(time.time() - init)
    # print(res.shape)