import cv2 as cv
import numpy as np
from skimage.measure import compare_psnr, compare_ssim, compare_mse

clahe = cv.createCLAHE(clipLimit=1., tileGridSize=(1, 1))  # Used in imadjust


def imadjust(img):
    return clahe.apply(img)


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


def mse(imgs, refs):

    return compare_mse(im1=imgs, im2=refs)