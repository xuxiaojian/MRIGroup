import numpy as np
import scipy.io as sio
from skimage.util import view_as_windows


def LiverReadValidDataset(root_path):
    print('[dataLoader.py] Load SR Valid')

    mat_file = sio.loadmat(root_path + 'valid.mat')
    x_noised_valid = mat_file['x_noised_valid']
    x_sr_valid = mat_file['x_sr_valid']
    y_valid = mat_file['y_valid']

    for i in range(x_noised_valid.shape[0]):
        x_noised_valid[i, :, :] = ImageProcesseed(x_noised_valid[i, :, :])
        x_sr_valid[i, :, :] = ImageProcesseed(x_sr_valid[i, :, :])
        y_valid[i, :, :] = ImageProcesseed(y_valid[i, :, :])

    x_noised_valid.shape = [x_noised_valid.shape[0], x_noised_valid.shape[1], x_noised_valid.shape[2], 1]
    x_sr_valid.shape = [x_sr_valid.shape[0], x_sr_valid.shape[1], x_sr_valid.shape[2], 1]
    y_valid.shape = [y_valid.shape[0], y_valid.shape[1], y_valid.shape[2], 1]

    return x_noised_valid, x_sr_valid, y_valid


def LiverReadSRTrainDataset(root_path, cropped=False, windows_size=(100, 100), step=20):
    print('[dataLoader.py] Load SR Train')

    mat_file = sio.loadmat(root_path + 'sr_train.mat')
    x_noised = mat_file['x_noised']
    x_sr_train = mat_file['x_sr_train']
    y_sr_train = mat_file['y_sr_train']

    print('[dataLoader.py] Images Preprocess')
    for i in range(x_sr_train.shape[0]):
        x_noised[i, :, :] = ImageProcesseed(x_noised[i, :, :])
        x_sr_train[i, :, :] = ImageProcesseed(x_sr_train[i, :, :])
        y_sr_train[i, :, :] = ImageProcesseed(y_sr_train[i, :, :])

    if cropped is True:
        x_sr_trains = view_as_windows(x_sr_train[0, :, :].copy(), windows_size, step)
        batch_size = x_sr_train.shape[0]
        height_patches = x_sr_trains.shape[0]
        width_patches = x_sr_trains.shape[1]

        print('[dataLoader.py] Images Crop, Amount: ', height_patches * width_patches * batch_size)
        x_noised_patches = np.zeros([
            height_patches * width_patches * batch_size, windows_size[0], windows_size[1]])
        x_sr_train_patches = np.zeros(
            [height_patches * width_patches * batch_size, windows_size[0], windows_size[1]])
        y_sr_train_patches = np.zeros(
            [height_patches * width_patches * batch_size, windows_size[0], windows_size[1]])

        count = 0
        for i in range(batch_size):
            x_noiseds = view_as_windows(x_noised[i, :, :], windows_size, step)
            x_sr_trains = view_as_windows(x_sr_train[i, :, :], windows_size, step)
            y_sr_trains = view_as_windows(y_sr_train[i, :, :], windows_size, step)

            for m in range(height_patches):
                for n in range(width_patches):
                    x_sr_train_patches[count, :, :] = x_sr_trains[m, n, :, :]
                    y_sr_train_patches[count, :, :] = y_sr_trains[m, n, :, :]
                    x_noised_patches[count, :, :] = x_noiseds[m, n, :, :]
                    count += 1

        x_noised = x_noised_patches
        x_sr_train = x_sr_train_patches
        y_sr_train = y_sr_train_patches

    print('[dataLoader.py] Convert Dimension')
    x_noised.shape = [x_noised.shape[0], x_noised.shape[1], x_noised.shape[2], 1]
    x_sr_train.shape = [x_sr_train.shape[0], x_sr_train.shape[1], x_sr_train.shape[2], 1]
    y_sr_train.shape = [y_sr_train.shape[0], y_sr_train.shape[1], y_sr_train.shape[2], 1]

    return x_noised, x_sr_train, y_sr_train


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
