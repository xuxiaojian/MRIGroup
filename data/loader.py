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

        mat_file = sio.loadmat(file_path[i] + '/NUFFT_' + scanlines + '.mat')
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

    config = configparser.ConfigParser()  # Load config file
    config.read('config.ini')

    from method.unet import Net3D
    import tensorflow as tf

    net = Net3D(config)

    model_path = config['3d-sr']['unet_model_path']
    batch_size = int(config['3d-sr']['unet_batch_size'])
    is_sr_concatenate = bool(int(config['data']['is_sr_concatenate']))

    x_unet = net.predict(x=x_src, y=y_src, batch_size=batch_size, model_path=model_path)
    x_unet = normalization(x_unet)

    x_imgs_unet = net.predict(x=x_imgs, y=y_imgs, batch_size=batch_size, model_path=model_path)
    x_imgs_unet = normalization(x_imgs_unet)

    tf.reset_default_graph()

    if is_sr_concatenate:
        x_src = np.concatenate([x_unet, x_src], axis=-1)
        x_imgs = np.concatenate([x_imgs_unet, x_imgs], axis=-1)
    else:
        x_src = x_unet
        x_imgs = x_imgs_unet

    def patch_operate(input_):
        batch, depth, _, _, channel = input_.shape
        result = view_as_windows(input_, (batch, depth, patch_size, patch_size, channel), patch_step)
        result = np.squeeze(result)
        result = np.ascontiguousarray(result)
        result.shape = (-1,) + result.shape[3:]

        if channel == 1:
            result = np.expand_dims(result, axis=-1)

        result = normalization(result)
        return result

    if is_patch:
        x_src = patch_operate(x_src); y_src=patch_operate(y_src)

        x_imgs = []; y_imgs = []
        for i in imgs_index:
            x_imgs.append(np.expand_dims(x_src[i], axis=0)); y_imgs.append(np.expand_dims(y_src[i], axis=0))

        x_imgs = np.concatenate(x_imgs, axis=0); y_imgs = np.concatenate(y_imgs, axis=0)

    return x_src, y_src, x_imgs, y_imgs


def source_sr_2d(root_path, mat_index, imgs_index, scanlines, is_patch, patch_size, patch_step):

    x_src, y_src, _, _ = source_sr_3d(root_path, mat_index, imgs_index, scanlines, is_patch, patch_size, patch_step)
    _, _, x_imgs, y_imgs = source_sr_3d(root_path, [mat_index[0]], imgs_index, scanlines, False, patch_size, patch_step)

    def user_reshape(imgs):

        batches, depth, height, width, channel = imgs.shape
        imgs = np.ascontiguousarray(imgs)
        imgs.shape = [batches * depth, height, width, channel]

        return imgs

    x_src = user_reshape(x_src); y_src = user_reshape(y_src)

    return x_src, y_src, x_imgs[:, 0, :, :, :], y_imgs[:, 0, :, :, :]


def mri_liver_crop(root_path, mat_index, scanlines):
    mask = [
        [116, 244, 70, 198],
        [96, 224, 51, 179],
        [169, 297, 111, 239],
        [121, 249, 101, 229],
        [111, 239, 93, 221],
        [106, 234, 103, 231],
        [96, 224, 98, 226],
        [51, 179, 103, 231],
        [76, 204, 86, 214],
    ]

    file_path = glob(root_path + '*_*')
    file_path.sort()

    x_src = []; y_src = []

    for i in mat_index:
        logging.info("Index of current mat file:[%d]. " % i + "Path: " + file_path[i])

        mat_file = sio.loadmat(file_path[i] + '/NUFFT_' + scanlines + '.mat')
        mat_data = mat_file['recon_MCNUFFT']
        mat_data = np.swapaxes(mat_data, 0, 2)
        mat_data = np.swapaxes(mat_data, 1, 3)
        mat_data = np.swapaxes(mat_data, 0, 1)
        x_src.append(mat_data[:, :, mask[i][0]:mask[i][1], mask[i][2]:mask[i][3]])

        mat_file = sio.loadmat(file_path[i] + '/CS_2000.mat')
        mat_data = mat_file['recon_CS']
        mat_data = np.swapaxes(mat_data, 0, 2)
        mat_data = np.swapaxes(mat_data, 1, 3)
        mat_data = np.swapaxes(mat_data, 0, 1)
        y_src.append(mat_data[:, :, mask[i][0]:mask[i][1], mask[i][2]:mask[i][3]])

    return x_src, y_src


def liver_crop_3d(root_path, mat_index, imgs_index, scanlines, is_patch, patch_size, patch_step):

    x_src, y_src = mri_liver_crop(root_path, mat_index, scanlines)

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


def liver_combine(root_path, mat_index, imgs_index, scanlines, is_patch, patch_size, patch_step):
    mask = [
        [116, 244, 70, 198],
        [96, 224, 51, 179],
        [169, 297, 111, 239],
        [121, 249, 101, 229],
        [111, 239, 93, 221],
        [106, 234, 103, 231],
        [96, 224, 98, 226],
        [51, 179, 103, 231],
        [76, 204, 86, 214],
    ]

    config = configparser.ConfigParser()  # Load config file
    config.read('config.ini')
    from method.unet import Net3D
    import tensorflow as tf

    x_src, y_src, _, _ = source_3d(root_path, mat_index, imgs_index, scanlines, 0, patch_size, patch_step)
    net = Net3D(config, input_shape=[10, 320, 320, 1], output_shape=[10, 320, 320, 1])
    model_path = config['3d-sr']['unet_model_path']
    batch_size = int(config['3d-sr']['unet_batch_size'])

    x_unet = net.predict(x=x_src, y=y_src, batch_size=batch_size, model_path=model_path)
    tf.reset_default_graph()

    x_src_liver, y_src_liver, _, _ = liver_crop_3d(root_path, mat_index, imgs_index, scanlines, is_patch, patch_size, patch_step)
    net = Net3D(config, input_shape=[10, 128, 128, 1], output_shape=[10, 128, 128, 1])
    model_path = config['data']['livercrop_model_path']
    batch_size = int(config['data']['livercrop_batch_size'])

    x_liver_unet = net.predict(x=x_src_liver, y=y_src_liver, batch_size=batch_size, model_path=model_path)
    tf.reset_default_graph()

    index = 0
    for i in mat_index:
        x_unet[index:index+96, :, mask[i][0]:mask[i][1], mask[i][2]:mask[i][3], :] = x_liver_unet[index:index+96, :, :, :, :]
        index += 96

    x_liver_combine = x_unet
    x_liver_combine = normalization(x_liver_combine)
    y_liver_combine = y_src

    x_imgs = []
    y_imgs = []
    for i in imgs_index:
        x_imgs.append(np.expand_dims(x_liver_combine[i], axis=0))
        y_imgs.append(np.expand_dims(y_liver_combine[i], axis=0))

    x_imgs = np.concatenate(x_imgs, axis=0)
    y_imgs = np.concatenate(y_imgs, axis=0)

    return x_liver_combine, y_liver_combine, x_imgs, y_imgs
