import numpy as np
from skimage.measure import compare_psnr, compare_ssim, compare_mse


def psnr(imgs, refs):
    height = imgs.shape[1]
    width = imgs.shape[2]
    value = []

    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, :].reshape([height, width])
        ref = refs[i, :, :, :].reshape([height, width])

        value.append(compare_psnr(im_true=ref, im_test=img, data_range=1))

    return np.array(value, dtype=np.float32)


def ssim(imgs, refs):
    height = imgs.shape[1]
    width = imgs.shape[2]
    value = []

    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, :].reshape([height, width])
        ref = refs[i, :, :, :].reshape([height, width])

        value.append(compare_ssim(X=ref, Y=img, data_range=1))

    return np.array(value, dtype=np.float32)


def loss(imgs, refs, type_='mse'):
    height = imgs.shape[1]
    width = imgs.shape[2]
    value = []

    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, :].reshape([height, width])
        ref = refs[i, :, :, :].reshape([height, width])
        if type_ == 'mse':
            value.append(compare_mse(im1=img, im2=ref))

    return np.array(value, dtype=np.float32)


def normalize(imgs):

    # 0-1 Normalization
    imgs -= np.amin(imgs)
    imgs /= np.amax(imgs)

    return imgs
