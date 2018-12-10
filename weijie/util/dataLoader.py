import numpy as np
import scipy.io as sio


def ReadValidDataset(data_index_read, root_path='/home/xiaojianxu/gan/data/mri/all/'):

    x = np.zeros(shape=[data_index_read.__len__(), 320, 320], dtype=np.float64)  # Features Data
    y = np.zeros(shape=[data_index_read.__len__(), 320, 320], dtype=np.float64)  # Labels Data

    print('[dataLoader.py] Load Valid .mat File in ' + root_path + '9.mat')

    mat_file = sio.loadmat(root_path + '9.mat')
    mat_x = mat_file['nims']
    mat_y = mat_file['cims']

    for i in range(data_index_read.__len__()):
        x[i, :, :] = mat_x[:, :, data_index_read[i]]
        y[i, :, :] = mat_y[:, :, data_index_read[i]]

        x[i, :, :] = ImageProcesseed(x[i, :, :])
        y[i, :, :] = ImageProcesseed(y[i, :, :])

    x.shape = [data_index_read.__len__(), 320, 320, 1]
    y.shape = [data_index_read.__len__(), 320, 320, 1]

    return x, y


def ReadRawDataset(mat_index_read, root_path='/home/xiaojianxu/gan/data/mri/all/'):

    x = np.zeros(shape=[mat_index_read.__len__() * 960, 320, 320], dtype=np.float64)  # Features Data
    y = np.zeros(shape=[mat_index_read.__len__() * 960, 320, 320], dtype=np.float64)  # Labels Data

    for i in range(mat_index_read.__len__()):

        print('[dataLoader.py] Load [%d] th ( %d Overall ) .mat File. ' % (i + 1, mat_index_read.__len__()),
              'Path: ' + root_path + str(mat_index_read[i]) + '.mat')

        mat_file = sio.loadmat(root_path + str(mat_index_read[i]) + '.mat')
        mat_x = mat_file['nims']
        mat_y = mat_file['cims']

        for j in range(960):
            x[i * 960 + j, :, :] = mat_x[:, :, j]
            y[i * 960 + j, :, :] = mat_y[:, :, j]

            x[i * 960 + j, :, :] = ImageProcesseed(x[i * 960 + j, :, :])
            y[i * 960 + j, :, :] = ImageProcesseed(y[i * 960 + j, :, :])

    x.shape = [mat_index_read.__len__() * 960, 320, 320, 1]
    y.shape = [mat_index_read.__len__() * 960, 320, 320, 1]

    return x, y


def ImageProcesseed(input_):

    # 0-1 Normalization
    input_ -= np.amin(input_)
    input_ /= np.amax(input_)

    return input_
