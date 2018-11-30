import numpy as np
import scipy.io as sio
import tensorflow as tf
from util import image_util


def mat2nparray(index_read, root_path):

    # Convert .mat raw format to nparray format

    x = np.zeros(shape=[index_read.__len__() * 960, 320, 320])  # Features Data
    y = np.zeros(shape=[index_read.__len__() * 960, 320, 320])  # Labels Data

    for i in range(index_read.__len__()):
        print('Loading: [%d] th ( %d Overall ) .mat File' % (i + 1, index_read.__len__()),
              'MAT Path: ' + root_path + str(index_read[i]) + '.mat')

        mat_file = sio.loadmat(root_path + str(index_read[i]) + '.mat')
        mat_x = mat_file['nims']
        mat_y = mat_file['cims']

        for j in range(960):
            x[i * 960 + j, :, :] = mat_x[:, :, j]
            y[i * 960 + j, :, :] = mat_y[:, :, j]

    x.shape = [index_read.__len__() * 960, 320, 320, 1]
    y.shape = [index_read.__len__() * 960, 320, 320, 1]

    return x, y


def read_mnist(num_train, num_test):

    # Easy Data For Debugging Network

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


