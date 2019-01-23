import tensorflow as tf
from tensorflow import keras as keras
import os
import shutil
from tensorboardX import SummaryWriter
from util import img_process
import scipy.io as sio


# Clean output folder
def new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


# Custom metrics for Kearas
def psnr_metrics(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))


def ssim_metrics(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))


# Custom callback for Kearas
class KearsCallback(keras.callbacks.Callback):
    def __init__(self, imgs_data_train, imgs_label_train, imgs_data_valid, imgs_label_valid, path, config_info):
        super().__init__()

        self.imgs_data_train = imgs_data_train
        self.imgs_label_train = imgs_label_train
        self.imgs_data_valid = imgs_data_valid
        self.imgs_label_valid = imgs_label_valid

        self.path = path
        self.val_path = self.path + 'val/'
        new_folder(self.val_path)

        self.writer = TBXWriter(path, config_info)

    def on_epoch_end(self, epoch, logs=None):
        imgs_predict_train = self.model.predict(self.imgs_data_train)
        imgs_predict_valid = self.model.predict(self.imgs_data_valid)

        self.writer.train_epoch(loss=logs['loss'], psnr=logs['psnr_metrics'], ssim=logs['psnr_metrics'], step=self.step)
        self.writer.valid_epoch(loss=logs['val_loss'], psnr=logs['val_psnr_metrics'], ssim=logs['val_psnr_metrics'],
                                step=self.step)

        self.writer.imgs_train_epoch(imgs_predict_train, self.imgs_label_train, step=self.step)
        self.writer.imgs_valid_epoch(imgs_predict_valid, self.imgs_label_valid, step=self.step)
        sio.savemat(self.val_path + '%d.mat' % self.step, {'predict': imgs_predict_valid})

    def on_batch_end(self, batch, logs=None):
        self.writer.train_batch(loss=logs['loss'], psnr=logs['psnr_metrics'], ssim=logs['psnr_metrics'], step=self.step)
        self.step = self.step + 1

    def on_train_begin(self, logs=None):
        self.step = 0
        self.writer.imgs_train_init(self.imgs_data_train, self.imgs_label_train)
        self.writer.imgs_valid_init(self.imgs_data_valid, self.imgs_label_valid)
        sio.savemat(self.val_path + 'init.mat', {'data': self.imgs_data_valid, 'label': self.imgs_label_valid})


class TBXWriter(object):

    def __init__(self, path, config_info):
        self.writer = SummaryWriter(path)
        self.writer.add_text(tag='config', text_string=config_info, global_step=0)

    def train_batch(self, loss, psnr, ssim, step):
        self.writer.add_scalar(tag='train_batch/loss', scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag='train_batch/psnr', scalar_value=psnr, global_step=step)
        self.writer.add_scalar(tag='train_batch/ssim', scalar_value=ssim, global_step=step)

    def train_epoch(self, loss, psnr, ssim, step):
        self.writer.add_scalar(tag='train_epoch/loss', scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag='train_epoch/psnr', scalar_value=psnr, global_step=step)
        self.writer.add_scalar(tag='train_epoch/ssim', scalar_value=ssim, global_step=step)

    def valid_epoch(self, loss, psnr, ssim, step):
        self.writer.add_scalar(tag='valid_epoch/loss', scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag='valid_epoch/psnr', scalar_value=psnr, global_step=step)
        self.writer.add_scalar(tag='valid_epoch/ssim', scalar_value=ssim, global_step=step)

    def imgs_train_epoch(self, imgs, y_imgs, step):
        loss = img_process.loss(imgs, y_imgs, type_='mse')
        psnr = img_process.psnr(imgs, y_imgs)
        ssim = img_process.ssim(imgs, y_imgs)

        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            self.writer.add_scalar(tag='imgs_train/index%d_loss' % i, scalar_value=loss[i], global_step=step)
            self.writer.add_scalar(tag='imgs_train/index%d_psnr' % i, scalar_value=psnr[i], global_step=step)
            self.writer.add_scalar(tag='imgs_train/index%d_ssim' % i, scalar_value=ssim[i], global_step=step)
            for j in range(channel):
                self.writer.add_image(tag='train/index%d_chaneel%d_predict' % (i, j),
                                      img_tensor=img_process.normalize(imgs[i].reshape([width_img, height_img])),
                                      global_step=step, dataformats='HW')
        pass

    def imgs_train_init(self, x_imgs, y_imgs):

        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]
        channel = x_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='train/index%d_chaneel%d_data' % (i, j),
                                      img_tensor=img_process.normalize(x_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
                self.writer.add_image(tag='train/index%d_chaneel%d_label' % (i, j),
                                      img_tensor=img_process.normalize(y_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')

    def imgs_valid_epoch(self, imgs, y_imgs, step):

        loss = img_process.loss(imgs, y_imgs, type_='mse')
        psnr = img_process.psnr(imgs, y_imgs)
        ssim = img_process.ssim(imgs, y_imgs)

        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            self.writer.add_scalar(tag='imgs_valid/index_%d_loss' % i, scalar_value=loss[i], global_step=step)
            self.writer.add_scalar(tag='imgs_valid/index_%d_psnr' % i, scalar_value=psnr[i], global_step=step)
            self.writer.add_scalar(tag='imgs_valid/index_%d_ssim' % i, scalar_value=ssim[i], global_step=step)
            for j in range(channel):
                self.writer.add_image(tag='valid/index%d_chaneel%d_predict' % (i, j),
                                      img_tensor=img_process.normalize(imgs[i].reshape([width_img, height_img])),
                                      global_step=step, dataformats='HW')

    def imgs_valid_init(self, x_imgs, y_imgs):
        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]
        channel = x_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='valid/index%d_chaneel%d_data' % (i, j),
                                      img_tensor=img_process.normalize(x_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
                self.writer.add_image(tag='valid/index%d_chaneel%d_label' % (i, j),
                                      img_tensor=img_process.normalize(y_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
