import numpy as np
import scipy.io as sio


def train_sr(root_path):
    index = [5, 6, 7, 8]
    num_imgs = 960

    print('Begin Loading Train Data....')
    height = 320
    width = 320
    size_batch = num_imgs * index.__len__()
    x_train = np.zeros([size_batch, height, width])
    y_train = np.zeros([size_batch, height, width])

    count = 0
    index_count = 1
    for i in index:
        print('[%d] th, [%d] Total' % (index_count, index.__len__()))
        index_count = index_count + 1
        print('The File Path:' + root_path + 'mat/' + str(i) + '.mat')
        mats = sio.loadmat(root_path + 'mat/' + str(i) + '.mat')
        nims = mats['nims']
        cims = mats['cims']

        for j in range(num_imgs):
            x_train[count, :, :] = nims[:, :, j]
            y_train[count, :, :] = cims[:, :, j]
            count = count + 1

    x_train.shape = [size_batch, height, width, 1]
    y_train.shape = [size_batch, height, width, 1]

    # Obtain Validation Data
    print('Begin Loading Validation Data....')
    print('The File Path:' + root_path + 'mat/9.mat')
    mats = sio.loadmat(root_path + 'mat/9.mat')
    nims = mats['nims']
    cims = mats['cims']

    index = [10, 15, 20, 25, 30, 35]
    height, width, _ = nims.shape
    size_batch = index.__len__()

    x_val = np.zeros([size_batch, height, width])
    y_val = np.zeros([size_batch, height, width])

    for i in range(size_batch):
        x_val[i, :, :] = nims[:, :, index[i]]
        y_val[i, :, :] = cims[:, :, index[i]]

    x_val.shape = [size_batch, height, width, 1]
    y_val.shape = [size_batch, height, width, 1]

    return x_train, y_train, x_val, y_val


def train_unet(root_path):
    index = [1]

    print('Begin Loading Train Data....')
    x_train = []
    y_train = []

    # count = 0
    index_count = 1
    for i in index:
        print('[%d] th, [%d] Total' % (index_count, index.__len__()))
        index_count = index_count + 1
        print('The File Path:' + root_path + 'mat/' + str(i) + '.mat')
        mats = sio.loadmat(root_path + 'mat/' + str(i) + '.mat')
        nims = mats['nims']
        cims = mats['cims']

        num_imgs = nims.shape[0]
        for j in range(num_imgs):
            x_train.append(nims[j, :, :])
            y_train.append(cims[j, :, :])
            # count = count + 1

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train.shape = [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1]
    y_train.shape = [y_train.shape[0], y_train.shape[1], y_train.shape[2], 1]

    # Obtain Validation Data
    print('Begin Loading Validation Data....')
    print('The File Path:' + root_path + 'mat/9.mat')
    mats = sio.loadmat(root_path + 'mat/9.mat')
    nims = mats['nims']
    cims = mats['cims']

    index = [10, 15, 20, 25, 30, 35]
    _, height, width = nims.shape
    size_batch = index.__len__()

    x_val = np.zeros([size_batch, height, width])
    y_val = np.zeros([size_batch, height, width])

    for i in range(size_batch):
        x_val[i, :, :] = nims[index[i], :, :]
        y_val[i, :, :] = cims[index[i], :, :]

    x_val.shape = [size_batch, height, width, 1]
    y_val.shape = [size_batch, height, width, 1]

    return x_train, y_train, x_val, y_val


# noinspection PyTupleAssignmentBalance
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

    def _fix_batch(self, n):
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
