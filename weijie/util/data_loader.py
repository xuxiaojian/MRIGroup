import numpy as np
import scipy.io as sio
from util.img_process import normalize
from skimage.util import view_as_windows


def load_old_unet_train(root_path):
    root_path = root_path + 'old/'

    # ####################
    img = sio.loadmat(root_path + 'unet_train.mat')
    x_train = img['x_unet_train']
    y_train = img['y_unet_train']

    batch_size, width, height = x_train.shape
    for i in range(batch_size):
        x_train[i] = normalize(x_train[i])
        y_train[i] = normalize(y_train[i])

    x_train.shape = [batch_size, width, height, 1]
    y_train.shape = [batch_size, width, height, 1]

    x_train_imgs = x_train[2].reshape([1, width, height, 1])
    y_train_imgs = y_train[2].reshape([1, width, height, 1])

    # ####################
    img = sio.loadmat(root_path + 'valid.mat')
    x_val = img['x_noised_valid']
    y_val = img['y_valid']

    batch_size, width, height = x_val.shape
    for i in range(batch_size):
        x_val[i] = normalize(x_val[i])
        y_val[i] = normalize(y_val[i])

    x_val.shape = [batch_size, width, height, 1]
    y_val.shape = [batch_size, width, height, 1]

    x_val_imgs = x_val[:2]
    y_val_imgs = y_val[:2]

    return x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs


def load_old_sr_train(root_path):
    root_path = root_path + 'old/'

    # ####################
    img = sio.loadmat(root_path + 'sr_train.mat')
    x_train_feature1 = img['x_noised']
    x_train_feature2 = img['x_sr_train']
    y_train = img['y_sr_train']

    batch_size, width, height = x_train_feature1.shape
    for i in range(batch_size):
        x_train_feature1[i] = normalize(x_train_feature1[i])
        x_train_feature2[i] = normalize(x_train_feature2[i])
        y_train[i] = normalize(y_train[i])

    x_train_feature1.shape = [batch_size, width, height, 1]
    x_train_feature2.shape = [batch_size, width, height, 1]
    y_train.shape = [batch_size, width, height, 1]

    x_train_imgs_feature1 = x_train_feature1[2].reshape([1, width, height, 1])
    x_train_imgs_feature2 = x_train_feature2[2].reshape([1, width, height, 1])
    y_train_imgs = y_train[2].reshape([1, width, height, 1])

    # ####################
    img = sio.loadmat(root_path + 'valid.mat')
    x_val_feature1 = img['x_noised_valid']
    x_val_feature2 = img['x_sr_valid']
    y_val = img['y_valid']

    batch_size, width, height = x_val_feature1.shape
    for i in range(batch_size):
        x_val_feature1[i] = normalize(x_val_feature1[i])
        x_val_feature2[i] = normalize(x_val_feature2[i])
        y_val[i] = normalize(y_val[i])

    x_val_feature1.shape = [batch_size, width, height, 1]
    x_val_feature2.shape = [batch_size, width, height, 1]
    y_val.shape = [batch_size, width, height, 1]

    x_val_imgs_feature1 = x_val_feature1[:2]
    x_val_imgs_feature2 = x_val_feature2[:2]
    y_val_imgs = y_val[:2]

    return x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
        y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, y_val_imgs


def load_xiaojian_sr_train(root_path):
    # Jan 30
    root_path = root_path + 'xiaojian_jan30/'

    # ####################
    img = sio.loadmat(root_path + 'test_x.mat')
    x_train_feature1 = img['img']
    img = sio.loadmat(root_path + 'test_y.mat')
    y_train = img['img']
    img = sio.loadmat(root_path + 'test_premat.mat')
    x_train_feature2 = img['img']

    batch_size, width, height, chaneel = x_train_feature1.shape
    for i in range(batch_size):
        x_train_feature1[i] = normalize(x_train_feature1[i])
        x_train_feature2[i] = normalize(x_train_feature2[i])
        y_train[i] = normalize(y_train[i])

    x_train_feature1.shape = [batch_size, width, height, 1]
    x_train_feature2.shape = [batch_size, width, height, 1]
    y_train.shape = [batch_size, width, height, 1]

    x_train_imgs_feature1 = x_train_feature1[2].reshape([1, width, height, 1])
    x_train_imgs_feature2 = x_train_feature2[2].reshape([1, width, height, 1])
    y_train_imgs = y_train[2].reshape([1, width, height, 1])

    # ####################
    img = sio.loadmat(root_path + 'val_x.mat')
    x_val_feature1 = img['img']
    img = sio.loadmat(root_path + 'val_y.mat')
    y_val = img['img']
    img = sio.loadmat(root_path + 'val_premat.mat')
    x_val_feature2 = img['img']

    batch_size, width, height, chaneel = x_val_feature1.shape
    for i in range(batch_size):
        x_val_feature1[i] = normalize(x_val_feature1[i])
        x_val_feature2[i] = normalize(x_val_feature2[i])
        y_val[i] = normalize(y_val[i])

    x_val_feature1.shape = [batch_size, width, height, 1]
    x_val_feature2.shape = [batch_size, width, height, 1]
    y_val.shape = [batch_size, width, height, 1]

    x_val_imgs_feature1 = x_val_feature1[15:16]
    x_val_imgs_feature2 = x_val_feature2[15:16]
    y_val_imgs = y_val[15:16]

    return x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
        y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, y_val_imgs


# noinspection PyUnresolvedReferences
def mat2numpy(root_path, type_, index_mat, index_images, debug=False):
    ####################################################
    #        Convert .Mat to Images(Numpy Format)      #
    #                                                  #
    # If necessary, you can change it to show your     #
    # custom method of data loading.                   #
    ####################################################
    data = []
    ground_truth = []

    for i in range(index_mat.__len__()):
        path_file = root_path + type_ + '/' + str(index_mat[i]) + '.mat'
        print('[%d] th, [%d] Total. ' % (i+1, index_mat.__len__()) + 'The Path of File:' + path_file)

        mats = sio.loadmat(path_file)
        nims = mats['nims']
        cims = mats['cims']

        num_images = None
        if type_ == 'mri_healthy_liver':
            num_images = nims.shape[0]
        if type_ == 'mri_healthy_all':
            num_images = nims.shape[2]

        for j in range(num_images):
            if type_ == 'mri_healthy_liver':
                data.append(normalize(nims[j, :, :]))
                ground_truth.append(normalize(cims[j, :, :]))
            if type_ == 'mri_healthy_all':
                data.append(normalize(nims[:, :, j]))
                ground_truth.append(normalize(cims[:, :, j]))

    data = np.array(data)
    ground_truth = np.array(ground_truth)

    size_batch = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]

    data.shape = [size_batch, width, height, 1]
    ground_truth.shape = [size_batch, width, height, 1]

    data_images = np.zeros(shape=[index_images.__len__(), width, height, 1])
    ground_truth_images = np.zeros(shape=[index_images.__len__(), width, height, 1])

    for i in range(index_images.__len__()):
        data_images[i] = data[index_images[i]]
        ground_truth_images[i] = ground_truth[index_images[i]]

    if debug:
        data = data[:10]
        ground_truth = ground_truth[:10]

    return data, ground_truth, data_images, ground_truth_images


def crop_images(imgs, window_size, step):
    batch_size = imgs.shape[0]
    channel = imgs.shape[3]

    crop_imgs = view_as_windows(imgs, window_shape=[batch_size, window_size, window_size, channel], step=step)

    crop_num_width = crop_imgs.shape[1]
    crop_num_height = crop_imgs.shape[2]

    crop_imgs = np.ascontiguousarray(crop_imgs, dtype=np.float64)
    crop_imgs.shape = [crop_num_width * crop_num_height * batch_size, window_size, window_size, channel]

    return crop_imgs
