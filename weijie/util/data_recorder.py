import numpy as np
import scipy.io as sio
import scipy.misc as smisc
import matplotlib.pyplot as plt


class VisualinPyplot(object):

    def __init__(self, name, output_path, index_fig):

        self.index_fig = index_fig
        self.data = []
        self.output_path = output_path
        self.name = name

    def add(self, newdata):

        self.data.append(newdata)
        plt.figure(self.index_fig)
        plt.plot(self.data)
        plt.title(self.name)
        plt.savefig(self.output_path + self.name + '.png')


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img


def save_mat(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """

    sio.savemat(path, {'img': img})


def save_img(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    img = to_rgb(img)
    smisc.imsave(path, img.round().astype(np.uint8))
