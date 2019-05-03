import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from methods import utilities
import scipy.io as sio


def psnr_tf(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)


def ssim_tf(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)


class KerasCallBack(tf.keras.callbacks.Callback):
    def __init__(self, output_path, train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset, config_info=None):
        super().__init__()
        self.writer = SummaryWriter(output_path)
        self.global_batch = 0

        self.validation_path = output_path + 'validation/'
        utilities.new_folder(self.validation_path)

        if config_info is not None:
            self.writer.add_text(tag='config', text_string=config_info, global_step=0)

        with tf.Session() as sess:
            self.train_x, self.train_y = sess.run(train_dataset.make_one_shot_iterator().get_next())
            self.valid_x, self.valid_y = sess.run(valid_dataset.make_one_shot_iterator().get_next())

    def on_train_begin(self, logs=None):
        sio.savemat(self.validation_path + 'init.mat', {'x': self.valid_x, 'y': self.valid_y})

        for i in range(self.train_x.shape[0]):
            self.add_imgs(self.train_x[i], tag='train/%d_x' % i, step=0)
            self.add_imgs(self.train_y[i], tag='train/%d_y' % i, step=0)

            self.add_imgs(self.valid_x[i], tag='valid/%d_x' % i, step=0)
            self.add_imgs(self.valid_y[i], tag='valid/%d_y' % i, step=0)

    def on_train_batch_end(self, batch, logs=None):
        self.writer.add_scalar(tag='train_batch/loss', scalar_value=logs['loss'], global_step=self.global_batch)
        self.writer.add_scalar(tag='train_batch/psnr', scalar_value=logs['psnr_tf'], global_step=self.global_batch)
        self.writer.add_scalar(tag='train_batch/ssim', scalar_value=logs['ssim_tf'], global_step=self.global_batch)
        self.global_batch += 1

    def on_epoch_end(self, epoch, logs=None):
        self.writer.add_scalar(tag='train_epoch/loss', scalar_value=logs['loss'], global_step=epoch)
        self.writer.add_scalar(tag='train_epoch/psnr', scalar_value=logs['psnr_tf'], global_step=epoch)
        self.writer.add_scalar(tag='train_epoch/ssim', scalar_value=logs['ssim_tf'], global_step=epoch)

        self.writer.add_scalar(tag='valid_epoch/loss', scalar_value=logs['val_loss'], global_step=epoch)
        self.writer.add_scalar(tag='valid_epoch/psnr', scalar_value=logs['val_psnr_tf'], global_step=epoch)
        self.writer.add_scalar(tag='valid_epoch/ssim', scalar_value=logs['val_ssim_tf'], global_step=epoch)

        train_pre = self.model.predict(self.train_x)
        valid_pre = self.model.predict(self.valid_x)
        sio.savemat(self.validation_path + 'epoch.%d.mat' % epoch, {'predict': valid_pre})

        for i in range(train_pre.shape[0]):
            self.add_imgs(train_pre[i], tag='train/%d_predict' % i, step=epoch)
            self.add_imgs(valid_pre[i], tag='valid/%d_predict' % i, step=epoch)

    def add_imgs(self, imgs_input, tag, step):
        from skimage.color import gray2rgb

        channel = imgs_input.shape[-1]
        width = imgs_input.shape[-2]
        height = imgs_input.shape[-3]

        imgs_input.shape = [-1, height, width, channel]
        new_batch = imgs_input.shape[0]

        assert channel == 1 or channel == 3

        if channel == 1:
            imgs_output = np.zeros(shape=[new_batch, height, width, 3])
            for i in range(new_batch):
                imgs_output[i] = gray2rgb(np.squeeze(imgs_input[i]))
        else:
            imgs_output = imgs_input

        self.writer.add_images(tag=tag, img_tensor=imgs_output, global_step=step, dataformats='NHWC')
