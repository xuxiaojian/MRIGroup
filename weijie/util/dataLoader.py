import numpy as np
import scipy.io as sio
from util import imgProcess


def train_sr(root_path):

    # Obtain Traning Data
    index = [2, 3, 4, 5, 6, 7, 8]
    num_imgs = 960

    # For Test Code
    # index = [2]
    # num_imgs = 70

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
            x_train[count, :, :] = imgProcess.normalize(nims[:, :, j])
            y_train[count, :, :] = imgProcess.normalize(cims[:, :, j])
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
        x_val[i, :, :] = imgProcess.normalize(nims[:, :, index[i]])
        y_val[i, :, :] = imgProcess.normalize(cims[:, :, index[i]])

    x_val.shape = [size_batch, height, width, 1]
    y_val.shape = [size_batch, height, width, 1]

    return x_train, y_train, x_val, y_val


def train_unet(root_path):
    x_train = np.load(root_path + 'x_tra_unet.npy')
    y_train = np.load(root_path + 'y_tra_unet.npy')
    x_val = np.load(root_path + 'x_val.npy')
    y_val = np.load(root_path + 'y_val.npy')

    return x_train, y_train, x_val, y_val


def _gen_train_unet(root_path):
    mats = sio.loadmat(root_path + 'mat/1.mat')
    nims = mats['nims']
    cims = mats['cims']

    height, width, size_batch = nims.shape
    x = np.zeros([size_batch, height, width])
    y = np.zeros([size_batch, height, width])

    for i in range(size_batch):
        x[i, :, :] = imgProcess.normalize(nims[:, :, i])
        y[i, :, :] = imgProcess.normalize(cims[:, :, i])

    x.shape = [size_batch, height, width, 1]
    y.shape = [size_batch, height, width, 1]

    np.save(root_path + 'x_tra_unet.npy', x)
    np.save(root_path + 'y_tra_unet.npy', y)
