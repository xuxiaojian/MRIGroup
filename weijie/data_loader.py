import numpy as np
import scipy.io as sio
from scadec import image_util


def get_unet_train(unet_train_index, root_path):
    unet_train_x = np.zeros(shape=[unet_train_index.__len__() * 960, 320, 320])
    unet_train_y = np.zeros(shape=[unet_train_index.__len__() * 960, 320, 320])

    for i in range(unet_train_index.__len__()):
        print('Loading: %d th ( %d Overall ) [UNet Train Dataset] ...' % (i + 1, unet_train_index.__len__()))
        print('MAT Path: ' + root_path + str(unet_train_index[i]) + '.mat')

        mat_data = sio.loadmat(root_path + str(unet_train_index[i]) + '.mat')
        mat_x = mat_data['nims']
        mat_y = mat_data['cims']

        for j in range(960):
            unet_train_x[i * 960 + j, :, :] = mat_x[:, :, j]
            unet_train_y[i * 960 + j, :, :] = mat_y[:, :, j]

    unet_train_x.shape = [unet_train_index.__len__() * 960, 320, 320, 1]
    unet_train_y.shape = [unet_train_index.__len__() * 960, 320, 320, 1]

    return image_util.SimpleDataProvider(unet_train_x, unet_train_y)


def get_srnet_train(srnet_train_index, root_path):
    srnet_train_x = np.zeros(shape=[srnet_train_index.__len__() * 960, 320, 320])
    srnet_train_y = np.zeros(shape=[srnet_train_index.__len__() * 960, 320, 320])

    for i in range(srnet_train_index.__len__()):
        print('Loading: %d th ( %d Overall ) [SRNet Train Dataset] ...' % (i + 1, srnet_train_index.__len__()))
        print('MAT Path: ' + root_path + str(srnet_train_index[i]) + '.mat')

        mat_data = sio.loadmat(root_path + str(srnet_train_index[i]) + '.mat')
        mat_x = mat_data['nims']
        mat_y = mat_data['cims']

        for j in range(960):
            srnet_train_x[i * 960 + j, :, :] = mat_x[:, :, j]
            srnet_train_y[i * 960 + j, :, :] = mat_y[:, :, j]

    srnet_train_x.shape = [srnet_train_index.__len__() * 960, 320, 320, 1]
    srnet_train_y.shape = [srnet_train_index.__len__() * 960, 320, 320, 1]

    return image_util.SimpleDataProvider(srnet_train_x, srnet_train_y)


def get_test(test_index, root_path):
    test_x = np.zeros(shape=[test_index.__len__() * 960, 320, 320])
    test_y = np.zeros(shape=[test_index.__len__() * 960, 320, 320])

    for i in range(test_index.__len__()):
        print('Loading: %d th ( %d Overall ) [Test Dataset] ...' % (i + 1, test_index.__len__()))
        print('MAT Path: ' + root_path + str(test_index[i]) + '.mat')

        mat_data = sio.loadmat(root_path + str(test_index[i]) + '.mat')
        mat_x = mat_data['nims']
        mat_y = mat_data['cims']

        for j in range(960):
            test_x[i * 960 + j, :, :] = mat_x[:, :, j]
            test_y[i * 960 + j, :, :] = mat_y[:, :, j]

    test_x.shape = [test_index.__len__() * 960, 320, 320, 1]
    test_y.shape = [test_index.__len__() * 960, 320, 320, 1]

    return image_util.SimpleDataProvider(test_x, test_y)