from skimage.util import view_as_windows
import numpy as np
import logging
import os
import shutil


def patch(input_, patch_size, patch_step):
    result = view_as_windows(input_, (patch_size, patch_size), patch_step)
    patch_nums = result.shape[0] * result.shape[1]
    result = np.ascontiguousarray(result)
    result.shape = [patch_nums, patch_size, patch_size]

    return result


def normalization(imgs):
    if imgs.shape.__len__() == 5:
        batch, depth, width, height, channel = imgs.shape
        for i in range(batch):
            for k in range(depth):
                for j in range(channel):
                    imgs[i, k, :, :, j] -= np.amin(imgs[i, k, :, :, j])
                    amax = np.amax(imgs[i, k, :, :, j])
                    if amax != 0:
                        imgs[i, k, :, :, j] /= amax

    if imgs.shape.__len__() == 4:
        batch, width, height, channel = imgs.shape
        for i in range(batch):
            for j in range(channel):
                imgs[i, :, :, j] -= np.amin(imgs[i, :, :, j])
                amax = np.amax(imgs[i, :, :, j])
                if amax != 0:
                    imgs[i, :, :, j] /= amax

    if imgs.shape.__len__() == 3:
        width, height, channel = imgs.shape
        for j in range(channel):
            imgs[:, :, j] -= np.amin(imgs[:, :, j])
            amax = np.amax(imgs[:, :, j])
            if amax != 0:
                imgs[:, :, j] /= amax

    if imgs.shape.__len__() == 2:
        imgs -= np.amin(imgs[:, :])
        if np.amax(imgs[:, :]) != 0:
            imgs /= np.amax(imgs[:, :])

    return imgs


def set_logging(path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log_file = logging.FileHandler(filename=path + 'log.txt')
    log_file.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.root.addHandler(log_file)


# Clean output folder
def new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)