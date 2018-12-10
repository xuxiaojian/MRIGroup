import cv2 as cv
import numpy as np

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

        mse = np.mean(np.square(img - ref))
        value.append(10 * np.log10(1/mse))

    return np.array(value, dtype=np.float32)
