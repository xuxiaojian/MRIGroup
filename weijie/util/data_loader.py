import numpy as np
import scipy.io as sio
from util.img_process import normalize
from skimage.util import view_as_windows


# noinspection PyUnresolvedReferences
def mat2numpy(root_path, type_, index_mat, index_images):
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
