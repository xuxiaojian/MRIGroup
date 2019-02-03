import tensorflow as tf
from tensorflow import keras as keras
import os
import shutil
from tensorboardX import SummaryWriter
from data.loader import normalize
import scipy.io as sio


# Clean output folder
def new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


# Custom callback for Kearas
class KearsCallback(keras.callbacks.Callback):
    def __init__(self, imgs_data_train, imgs_label_train, imgs_data_valid, imgs_label_valid,
                 path, config_info, epoch_save):
        super().__init__()

        self.imgs_data_train = imgs_data_train
        self.imgs_label_train = imgs_label_train
        self.imgs_data_valid = imgs_data_valid
        self.imgs_label_valid = imgs_label_valid

        self.path = path
        self.epoch_save = epoch_save

        self.save_path = self.path + 'epoch_save/'
        new_folder(self.save_path)

        self.writer = TBXWriter(path, config_info)

    def on_epoch_end(self, epoch, logs=None):
        imgs_predict_train = self.model.predict(self.imgs_data_train)
        imgs_predict_valid = self.model.predict(self.imgs_data_valid)

        self.writer.train_epoch(loss=logs['loss'], psnr=logs['psnr_metrics'], ssim=logs['ssim_metrics'], step=self.step)
        self.writer.valid_epoch(loss=logs['val_loss'], psnr=logs['val_psnr_metrics'], ssim=logs['val_ssim_metrics'],
                                step=self.step)

        self.writer.imgs_train_epoch(imgs_predict_train, step=self.step)
        self.writer.imgs_valid_epoch(imgs_predict_valid, step=self.step)

        self.model.save_weights(self.path + 'final_weight.h5')

        if (epoch + 1) % self.epoch_save == 0:
            sio.savemat(self.save_path + 'epoch_%d.mat' % (epoch + 1), {'predict': imgs_predict_valid})
            self.model.save_weights(self.save_path + 'epoch_%d.h5' % (epoch + 1))

    def on_batch_end(self, batch, logs=None):
        self.writer.train_batch(loss=logs['loss'], psnr=logs['psnr_metrics'], ssim=logs['ssim_metrics'], step=self.step)
        self.step = self.step + 1

    def on_train_begin(self, logs=None):
        self.step = 0
        self.writer.imgs_train_init(self.imgs_data_train, self.imgs_label_train)
        self.writer.imgs_valid_init(self.imgs_data_valid, self.imgs_label_valid)

        sio.savemat(self.save_path + 'initial.mat', {'data': self.imgs_data_valid, 'label': self.imgs_label_valid})


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

    def imgs_train_epoch(self, imgs, step):
        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='train/index%d_predict_chaneel%d' % (i, j),
                                      img_tensor=normalize(imgs[i].reshape([width_img, height_img])),
                                      global_step=step, dataformats='HW')
        pass

    def imgs_train_init(self, x_imgs, y_imgs):

        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]

        channel_x = x_imgs.shape[3]
        channel_y = y_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel_x):
                self.writer.add_image(tag='train/index%d_data_chaneel%d' % (i, j),
                                      img_tensor=normalize(x_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
            for j in range(channel_y):
                self.writer.add_image(tag='train/index%d_label_chaneel%d' % (i, j),
                                      img_tensor=normalize(y_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')

    def imgs_valid_epoch(self, imgs, step):
        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='valid/index%d_predict_chaneel%d' % (i, j),
                                      img_tensor=normalize(imgs[i].reshape([width_img, height_img])),
                                      global_step=step, dataformats='HW')

    def imgs_valid_init(self, x_imgs, y_imgs):
        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]

        channel_x = x_imgs.shape[3]
        channel_y = y_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel_x):
                self.writer.add_image(tag='valid/index%d_data_chaneel%d' % (i, j),
                                      img_tensor=normalize(x_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
            for j in range(channel_y):
                self.writer.add_image(tag='valid/index%d_label_chaneel%d' % (i, j),
                                      img_tensor=normalize(y_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
