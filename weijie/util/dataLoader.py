import numpy as np
import scipy.io as sio
from skimage.util import view_as_windows


def AllUnetTrain(root_path):
    mats = sio.loadmat(root_path + 'mat/1.mat')
    nims = mats['nims']
    cims = mats['cims']

    height, width, size_batch = nims.shape
    x = np.zeros([size_batch, height, width])
    y = np.zeros([size_batch, height, width])

    for i in range(size_batch):
        x[i, :, :] = ImageProcesseed(nims[:, :, i])
        y[i, :, :] = ImageProcesseed(cims[:, :, i])

    x.shape = [size_batch, height, width, 1]
    y.shape = [size_batch, height, width, 1]

    return x, y


def AllValid(root_path):
    mats = sio.loadmat(root_path + 'mat/9.mat')
    nims = mats['nims']
    cims = mats['cims']

    index = [10, 15, 20, 25, 30, 35]
    height, width, _ = nims.shape
    size_batch = index.__len__()

    x = np.zeros([size_batch, height, width])
    y = np.zeros([size_batch, height, width])

    for i in range(size_batch):
        x[i, :, :] = ImageProcesseed(nims[:, :, index[i]])
        y[i, :, :] = ImageProcesseed(cims[:, :, index[i]])

    x.shape = [size_batch, height, width, 1]
    y.shape = [size_batch, height, width, 1]

    return x, y


def LiverValid(root_path):
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


def LiverSRTrain(root_path, cropped=False, size=(100, 100), step=20):
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
        x_sr_trains = view_as_windows(x_sr_train[0, :, :].copy(), size, step)
        batch_size = x_sr_train.shape[0]
        height_patches = x_sr_trains.shape[0]
        width_patches = x_sr_trains.shape[1]

        print('[dataLoader.py] Images Crop, Total Amount: ', height_patches * width_patches * batch_size)
        x_noised_patches = np.zeros([
            height_patches * width_patches * batch_size, size[0], size[1]])
        x_sr_train_patches = np.zeros(
            [height_patches * width_patches * batch_size, size[0], size[1]])
        y_sr_train_patches = np.zeros(
            [height_patches * width_patches * batch_size, size[0], size[1]])

        count = 0
        for i in range(batch_size):
            x_noiseds = view_as_windows(x_noised[i, :, :], size, step)
            x_sr_trains = view_as_windows(x_sr_train[i, :, :], size, step)
            y_sr_trains = view_as_windows(y_sr_train[i, :, :], size, step)

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


def ImageProcesseed(input_):

    # 0-1 Normalization
    input_ -= np.amin(input_)
    input_ /= np.amax(input_)

    return input_
