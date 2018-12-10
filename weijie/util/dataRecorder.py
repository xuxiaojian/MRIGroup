import matplotlib
import numpy as np
from util.imgProcess import imadjust

# For Linux-Compatible
matplotlib.use('agg')
import matplotlib.pyplot as plt

# 虽然我后续发现使用这个的话实时性是不如Tensorboard的，但是用来生成报告是极好的，当务之急应该是尽快**试下各种方法**才是王道！！！


class VisualPlot(object):

    def __init__(self):
        self.loss_train_epoch = []
        self.loss_valid_epoch = []

        self.psnr_train_epoch = []
        self.psnr_valid_epoch = []

        self.loss_train_batch = []
        self.psnr_train_batch = []

    def add_epoch(self, loss_train, loss_valid, psnr_train, psnr_valid):
        self.loss_train_epoch.append(loss_train)
        self.loss_valid_epoch.append(loss_valid)
        self.psnr_train_epoch.append(psnr_train)
        self.psnr_valid_epoch.append(psnr_valid)

    def add_batches(self, loss_train, psnr_train):
        self.loss_train_batch.append(loss_train)
        self.psnr_train_batch.append(psnr_train)

    def save(self, tofile=False, path=None):
        if tofile is True and path is None:
            print('[dataRecorder.py] If set tofile as True, then you also need to set up path')
            return 0

        plt.figure(1, figsize=(15, 4))
        plt.suptitle('Value of Loss in Epoch')
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_train_epoch)
        plt.title('Train')
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_valid_epoch)
        plt.title('Valid')
        if tofile is True:
            plt.savefig(path + 'ValueLossinEpoch.png')

        plt.figure(2, figsize=(15, 4))
        plt.suptitle('Value of PSNR in Epoch')
        plt.subplot(1, 2, 1)
        plt.plot(self.psnr_train_epoch)
        plt.title('Train')
        plt.subplot(1, 2, 2)
        plt.plot(self.psnr_valid_epoch)
        plt.title('Valid')
        if tofile is True:
            plt.savefig(path + 'ValuePSNRinEpoch.png')

        plt.figure(3, figsize=(15, 4))
        plt.suptitle('Batch')
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_train_batch)
        plt.title('Value of Loss')
        plt.subplot(1, 2, 2)
        plt.plot(self.psnr_train_batch)
        plt.title('Value of PSNR')
        if tofile is True:
            plt.savefig(path + 'Batch.png')

        if tofile is False:
            plt.show()


class VisualImage(object):

    def __init__(self, xs, ys, xs_noised=None):
        self.x = xs
        self.y = ys
        self.x_noised = xs_noised

        self.num_data = self.x.shape[0]
        self.height = self.x.shape[1]
        self.width = self.x.shape[2]
        self.channel = self.x.shape[3]

    def save(self, ys_pre, epoch, tofile=False, path=None, if_imadjust=False):

        for i in range(ys_pre.shape[0]):

            if tofile is True and path is None:
                print('[dataRecorder.py] If set tofile as True, then you also need to set up path')
                return 0

            if self.x_noised is not None:
                x_noised = (self.x_noised[i, :, :, :] * 255).reshape([self.height, self.width])
                x_noised[x_noised < 0] = 0
                x_noised[x_noised > 255] = 255

            x = (self.x[i, :, :, :] * 255).reshape([self.height, self.width])
            x[x < 0] = 0
            x[x > 255] = 255
            x = np.uint8(x)

            y_pre = (ys_pre[i, :, :, :] * 255).reshape([self.height, self.width])
            y_pre[y_pre < 0] = 0
            y_pre[y_pre > 255] = 255
            y_pre = np.uint8(y_pre)

            y = (self.y[i, :, :, :] * 255).reshape([self.height, self.width])
            y[y < 0] = 0
            y[y > 255] = 255
            y = np.uint8(y)

            if if_imadjust:
                x = imadjust(x)
                y_pre = imadjust(y_pre)
                y = imadjust(y)

            if self.x_noised is not None:
                plt.figure(1, figsize=(11, 3))
            else:
                plt.figure(1, figsize=(8, 3))
            plt.subplots_adjust(left=0.02, right=0.98)

            if self.x_noised is not None:
                plt.subplot(1, 4, 1)
                plt.imshow(x_noised, cmap='gray')
                plt.title('Noised X')
                plt.axis('off')
                plt.subplot(1, 4, 2)
                plt.imshow(x, cmap='gray')
                plt.title('X')
                plt.axis('off')
                plt.subplot(1, 4, 3)
                plt.imshow(y_pre, cmap='gray')
                plt.title('Predicted Y')
                plt.axis('off')
                plt.subplot(1, 4, 4)
                plt.imshow(y, cmap='gray')
                plt.title('Y')
                plt.axis('off')

            else:
                plt.subplot(1, 3, 1)
                plt.imshow(x, cmap='gray')
                plt.axis('off')
                plt.title('X')
                plt.subplot(1, 3, 2)
                plt.imshow(y_pre, cmap='gray')
                plt.axis('off')
                plt.title('Predicted Y')
                plt.subplot(1, 3, 3)
                plt.imshow(y, cmap='gray')
                plt.axis('off')
                plt.title('Y')

            if tofile is True:
                plt.savefig(path + str(epoch) + '_index_' + str(i) + '.png')

            if tofile is False:
                plt.show()
