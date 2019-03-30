from data.tools import patch, normalization
import numpy as np
import scipy.io as sio
from glob import glob
import logging


def mri_source(root_path, mat_index, scanlines):

    file_path = glob(root_path + '*_*')
    file_path.sort()

    x_src = []; y_src = []

    for i in mat_index:
        logging.info("Index of current mat file:[%d]. " % i + "Path: " + file_path[i])

        mat_file = sio.loadmat(file_path[i] + '/MCNUFFT_' + scanlines + '.mat')
        mat_data = mat_file['recon_MCNUFFT']
        mat_data = np.swapaxes(mat_data, 0, 2)
        mat_data = np.swapaxes(mat_data, 1, 3)
        mat_data = np.swapaxes(mat_data, 0, 1)
        x_src.append(mat_data)

        mat_file = sio.loadmat(file_path[i] + '/CS_2000.mat')
        mat_data = mat_file['recon_CS']
        mat_data = np.swapaxes(mat_data, 0, 2)
        mat_data = np.swapaxes(mat_data, 1, 3)
        mat_data = np.swapaxes(mat_data, 0, 1)
        y_src.append(mat_data)

    return x_src, y_src


def source_3d(root_path, mat_index, scanlines, type_, is_patch, patch_size, patch_step):

    x_src, y_src = mri_source(root_path, mat_index, scanlines)

    def operation(input_):
        result = np.concatenate(input_, axis=0)
        result = np.expand_dims(result, axis=-1)
        result = normalization(result)
        return result

    x_src = operation(x_src); y_src = operation(y_src)

    x_imgs = []; y_imgs = []

    x_imgs.append(np.expand_dims(x_src[10], axis=0)); y_imgs.append(np.expand_dims(y_src[10], axis=0))
    x_imgs.append(np.expand_dims(x_src[20], axis=0)); y_imgs.append(np.expand_dims(y_src[20], axis=0))
    x_imgs = np.concatenate(x_imgs, axis=0); y_imgs = np.concatenate(y_imgs, axis=0)

    return x_src, y_src, x_imgs, y_imgs
