import numpy as np
import scipy.io as sio
from scipy import misc
import tensorflow as tf


def read_mnist(num_train: "Number of Train Data", num_test: "Number of Test Data") \
        -> "Train Data and Test Data as SimpleDataProvider Object":

    # Easy Data For Debugging Network

    (img_train, _), (img_test, _) = tf.keras.datasets.mnist.load_data()
    # Only used source image in this problem

    if num_train - 1 > img_train.shape[0] or num_test - 1 > img_test.shape[0]:
        print("[data_loader: ]: Error - Mnist Dataset Index Excess, Max: %d in Train and %d in Test" %
              (img_train.shape[0], img_test.shape[0]))
        return None

    x_train = np.zeros([num_train, 28, 28])
    y_train = np.zeros([num_train, 28, 28])
    for i in range(num_train):
        y_train[i, :, :] = img_train[i, :, :]
        x_train[i, :, :] = misc.imresize(misc.imresize(img_train[i, :, :], [10, 10]), [28, 28])
    x_train.shape = [num_train, 28, 28, 1]
    y_train.shape = [num_train, 28, 28, 1]

    x_test = np.zeros([num_test, 28, 28])
    y_test = np.zeros([num_test, 28, 28])
    for i in range(num_test):
        y_test[i, :, :] = img_test[i, :, :]
        x_test[i, :, :] = misc.imresize(misc.imresize(img_test[i, :, :], [10, 10]), [28, 28])
    x_test.shape = [num_test, 28, 28, 1]
    y_test.shape = [num_test, 28, 28, 1]

    return SimpleDataProvider(data=x_train, truths=y_train), SimpleDataProvider(data=x_test, truths=y_test)


def read_matfiles(index_read, root_path):

    # Convert **customed** .mat raw format to nparray format

    x = np.zeros(shape=[index_read.__len__() * 960, 320, 320])  # Features Data
    y = np.zeros(shape=[index_read.__len__() * 960, 320, 320])  # Labels Data

    for i in range(index_read.__len__()):
        print('[data_loader: ]: [%d] th ( %d Overall ) .mat File' % (i + 1, index_read.__len__()),
              'MAT Path: ' + root_path + str(index_read[i]) + '.mat')

        mat_file = sio.loadmat(root_path + str(index_read[i]) + '.mat')
        mat_x = mat_file['nims']
        mat_y = mat_file['cims']

        for j in range(960):
            x[i * 960 + j, :, :] = mat_x[:, :, j]
            y[i * 960 + j, :, :] = mat_y[:, :, j]

    x.shape = [index_read.__len__() * 960, 320, 320, 1]
    y.shape = [index_read.__len__() * 960, 320, 320, 1]

    return SimpleDataProvider(data=x, truths=y)


# Following codes are copied from https://github.com/sunyumark/ScaDec-deep-learning-diffractive-tomography


# noinspection PyTupleAssignmentBalance,PyUnresolvedReferences
class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def __call__(self, n, fix=False):
        if type(n) == int and not fix:
            # X and Y are the images and truths
            train_data, truths = self._next_batch(n)
        elif type(n) == int and fix:
            train_data, truths = self._fix_batch(n)
        elif type(n) == str and n == 'full':
            train_data, truths = self._full_batch()
        else:
            raise ValueError("Invalid batch_size: " % n)

        return train_data, truths

    def _next_batch(self, n):
        pass

    def _full_batch(self):
        pass


class SimpleDataProvider(BaseDataProvider):

    def __init__(self, data, truths):
        super(SimpleDataProvider, self).__init__()
        self.data = np.float64(data)
        self.truths = np.float64(truths)
        self.img_channels = self.data[0].shape[2]
        self.truth_channels = self.truths[0].shape[2]
        self.file_count = data.shape[0]

    def _next_batch(self, n):
        idx = np.random.choice(self.file_count, n, replace=False)
        img = self.data[idx[0]]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))
        for i in range(n):
            X[i] = self._process_data(self.data[idx[i]])
            Y[i] = self._process_truths(self.truths[idx[i]])
        return X, Y

    def _fix_batch(self, n):
        # first n data
        img = self.data[0]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))
        for i in range(n):
            X[i] = self._process_data(self.data[i])
            Y[i] = self._process_truths(self.truths[i])
        return X, Y

    def _full_batch(self):
        return self.data, self.truths

    def _process_truths(self, truth):
        # normalization by channels
        truth = np.clip(np.fabs(truth), self.a_min, self.a_max)
        for channel in range(self.truth_channels):
            truth[:, :, channel] -= np.amin(truth[:, :, channel])
            truth[:, :, channel] /= np.amax(truth[:, :, channel])
        return truth

    def _process_data(self, data):
        # normalization by channels
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        for channel in range(self.img_channels):
            data[:, :, channel] -= np.amin(data[:, :, channel])
            data[:, :, channel] /= np.amax(data[:, :, channel])
        return data
