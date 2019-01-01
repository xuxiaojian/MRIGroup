import numpy as np
import scipy.io as sio
from util import imgProcess


def train_unet(root_path):
    x_train = np.load(root_path + 'x_tra_unet.npy')
    y_train = np.load(root_path + 'y_tra_unet.npy')
    x_val = np.load(root_path + 'x_val.npy')
    y_val = np.load(root_path + 'y_val.npy')

    return x_train, y_train, x_val, y_val


# Generlize Numpy Data for Loading Quickly
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


def _gen_valid_unet(root_path):
    mats = sio.loadmat(root_path + 'mat/9.mat')
    nims = mats['nims']
    cims = mats['cims']

    index = [10, 15, 20, 25, 30, 35]
    height, width, _ = nims.shape
    size_batch = index.__len__()

    x = np.zeros([size_batch, height, width])
    y = np.zeros([size_batch, height, width])

    for i in range(size_batch):
        x[i, :, :] = imgProcess.normalize(nims[:, :, index[i]])
        y[i, :, :] = imgProcess.normalize(cims[:, :, index[i]])

    x.shape = [size_batch, height, width, 1]
    y.shape = [size_batch, height, width, 1]

    np.save(root_path + 'x_val.npy', x)
    np.save(root_path + 'y_val.npy', y)
