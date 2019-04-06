from data.tools import normalization
import numpy as np
import scipy.io as sio
from glob import glob
import logging
from skimage.util import view_as_windows
import configparser


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


def source_3d(root_path, mat_index, imgs_index, scanlines, is_patch, patch_size, patch_step):

    x_src, y_src = mri_source(root_path, mat_index, scanlines)

    def operation(input_):
        result = np.concatenate(input_, axis=0)

        if is_patch:
            result = view_as_windows(result, (result.shape[0:2]) + (patch_size, patch_size), patch_step)
            result = np.squeeze(result)
            result = np.ascontiguousarray(result)
            result.shape = (-1, ) + result.shape[3:]

        result = np.expand_dims(result, axis=-1)
        result = normalization(result)
        return result

    x_src = operation(x_src); y_src = operation(y_src)

    x_imgs = []; y_imgs = []

    for i in imgs_index:
        x_imgs.append(np.expand_dims(x_src[i], axis=0)); y_imgs.append(np.expand_dims(y_src[i], axis=0))

    x_imgs = np.concatenate(x_imgs, axis=0); y_imgs = np.concatenate(y_imgs, axis=0)

    return x_src, y_src, x_imgs, y_imgs


def source_sr_3d(root_path, mat_index, imgs_index, scanlines, is_patch, patch_size, patch_step):

    x_src, y_src, x_imgs, y_imgs = source_3d(root_path, mat_index, imgs_index, scanlines, 0, patch_size, patch_step)
    # No patch first

    # x_src = x_src[:10]
    # y_src = y_src[:10]

    config = configparser.ConfigParser()  # Load config file
    config.read('config.ini')

    from method.unet import Net3D
    import tensorflow as tf

    net = Net3D(config)

    model_path = config['3d-sr']['unet_model_path']
    x_unet = net.predict(x=x_src, y=y_src, batch_size=8, model_path=model_path)
    x_unet = normalization(x_unet)

    x_imgs_unet = net.predict(x=x_imgs, y=y_imgs, batch_size=8, model_path=model_path)
    x_imgs_unet = normalization(x_imgs_unet)

    tf.reset_default_graph()

    x_src = np.concatenate([x_unet, x_src], axis=-1)
    x_imgs = np.concatenate([x_imgs_unet, x_imgs], axis=-1)

    return x_src, y_src, x_imgs, y_imgs
