import numpy as np
import scipy.io as sio


def trainNvalid(root_path, type_, index_train_mat, index_train_imgs, index_valid_mat, index_valid_imgs):

    if not(type_ == 'mri' or type_ == 'liver'):
        print('[ERROR]: WRONG TYPE')
        exit(0)

    ####################################################
    #                obtain Training Data              #
    ####################################################
    print('Begin Loading Train Data....')
    x_train = []
    y_train = []

    index_count = 1
    for i in index_train_mat:
        print('[%d] th, [%d] Total' % (index_count, index_train_mat.__len__()))
        index_count = index_count + 1

        print('The File Path:' + root_path + type_ + '/mat/' + str(i) + '.mat')
        mats = sio.loadmat(root_path + type_ + '/mat/' + str(i) + '.mat')
        nims = mats['nims']
        cims = mats['cims']

        if type_ == 'liver':
            num_imgs = nims.shape[0]
        else:
            num_imgs = nims.shape[2]

        for j in range(num_imgs):
            if type_ == 'liver':
                x_train.append(nims[j, :, :])
                y_train.append(cims[j, :, :])
            else:
                x_train.append(nims[:, :, j])
                y_train.append(cims[:, :, j])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train.shape = [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1]
    y_train.shape = [y_train.shape[0], y_train.shape[1], y_train.shape[2], 1]

    ####################################################
    #                obtain Validation Data            #
    ####################################################

    # Obtain Validation Data
    print('Begin Loading Validation Data....')

    x_val = []
    y_val = []

    index_count = 1
    for i in index_valid_mat:
        print('[%d] th, [%d] Total' % (index_count, index_valid_mat.__len__()))
        index_count = index_count + 1

        print('The File Path:' + root_path + type_ + '/mat/' + str(i) + '.mat')
        mats = sio.loadmat(root_path + type_ + '/mat/' + str(i) + '.mat')
        nims = mats['nims']
        cims = mats['cims']

        if type_ == 'liver':
            num_imgs = nims.shape[0]
        else:
            num_imgs = nims.shape[2]

        for j in range(num_imgs):
            if type_ == 'liver':
                x_val.append(nims[j, :, :])
                y_val.append(cims[j, :, :])
            else:
                x_val.append(nims[:, :, j])
                y_val.append(cims[:, :, j])

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    x_val.shape = [x_val.shape[0], x_val.shape[1], x_val.shape[2], 1]
    y_val.shape = [y_val.shape[0], y_val.shape[1], y_val.shape[2], 1]

    ####################################################
    #                     obtain  Imgs                 #
    ####################################################

    x_train_imgs = []
    y_train_imgs = []
    x_val_imgs = []
    y_val_imgs = []

    for i in index_train_imgs:
        x_train_imgs.append(x_train[i, :, :, :])
        y_train_imgs.append(y_train[i, :, :, :])

    for i in index_valid_imgs:
        x_val_imgs.append(x_val[i, :, :, :])
        y_val_imgs.append(y_val[i, :, :, :])

    x_train_imgs = np.array(x_train_imgs)
    y_train_imgs = np.array(y_train_imgs)
    x_val_imgs = np.array(x_val_imgs)
    y_val_imgs = np.array(y_val_imgs)

    return x_train, y_train, x_val, y_val, x_train_imgs, y_train_imgs, x_val_imgs, y_val_imgs
