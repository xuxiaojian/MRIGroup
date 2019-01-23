import numpy as np
import scipy.io as sio
from util.img_process import normalize


# noinspection PyUnresolvedReferences
def mat2numpy(root_path, type_, index_mat, index_images):
    ####################################################
    #        Convert .Mat to Images(Numpy Format)      #
    #                                                  #
    # If necessary, you can change it to show your     #
    # custom method of data loading.                   #
    ####################################################
    if not (type_ == 'mri' or type_ == 'liver'):
        print('[ERROR]: WRONG TYPE')
        exit(0)

    data = []
    ground_truth = []

    for i in range(index_mat.__len__()):
        path_file = root_path + type_ + '/mat/' + str(index_mat[i]) + '.mat'
        print('[%d] th, [%d] Total. ' % (i+1, index_mat.__len__()) + 'The Path of File:' + path_file)

        mats = sio.loadmat(path_file)
        nims = mats['nims']
        cims = mats['cims']

        if type_ == 'liver':
            num_images = nims.shape[0]
        else:
            num_images = nims.shape[2]

        for j in range(num_images):
            if type_ == 'liver':
                data.append(normalize(nims[j, :, :]))
                ground_truth.append(normalize(cims[j, :, :]))
            else:
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
