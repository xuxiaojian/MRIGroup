import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, \
    MaxPool2D, Conv2DTranspose, concatenate, Activation
from tensorflow.keras.optimizers import Adam
import shutil
import tensorflow as tf
import scipy.io as sio
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorboardX import SummaryWriter
from util import imgProcess
import numpy as np
import time
from util.dataLoader import ImageProcesseed


class KerasNetwork(object):

    def __init__(self, config):

        self.config = config

        self.set_parameter()
        self.set_data()

        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, max_val=1)

        def ssim(y_true, y_pred):
            return tf.image.ssim(y_pred, y_true, max_val=1)

        self.net_train = self.build_network(self.shape_batch)
        self.net_train.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr, ssim])

        self.net_train.summary()

    def train(self):

        config_info = self.config_info

        x_valid = self.valid_x
        y_valid = self.valid_y

        root_path = self.root_path

        epoch_save = self.epoch_save
        valid_path = self.valid_path

        class SaveValidation(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.tbwriter.add_scalar(tag='train/epoch_loss', scalar_value=logs['loss'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/epoch_psnr', scalar_value=logs['psnr'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/epoch_ssim', scalar_value=logs['ssim'], global_step=self.batch)

                pre_valid = self.model.predict(x_valid)
                mse_valid = imgProcess.mse(pre_valid, y_valid)
                psnr_valid = imgProcess.psnr(pre_valid, y_valid)
                ssim_valid = imgProcess.ssim(pre_valid, y_valid)

                self.tbwriter.add_scalar(tag='valid_avg/epoch_loss', scalar_value=mse_valid,
                                         global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_psnr', scalar_value=psnr_valid.mean(),
                                         global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_ssim', scalar_value=ssim_valid.mean(),
                                         global_step=self.batch)

                for i in range(psnr_valid.shape[0]):
                    self.tbwriter.add_scalar(tag='valid_%d/epoch_psnr' % i, scalar_value=psnr_valid[i],
                                             global_step=self.batch)
                    self.tbwriter.add_scalar(tag='valid_%d/epoch_ssim' % i, scalar_value=ssim_valid[i],
                                             global_step=self.batch)

                    pre_valid_ = pre_valid[i, :, :, :]
                    pre_valid_ = ImageProcesseed(pre_valid_)

                    self.tbwriter.add_image(tag='valid_%d/predict' % i,
                                            img_tensor=np.reshape(pre_valid_, [320, 320]),
                                            global_step=self.batch)

                if (epoch + 1) % epoch_save == 0:
                    sio.savemat(valid_path + '%d.mat' % (epoch + 1), {
                        'pre_valid': pre_valid,
                    })

            def on_batch_end(self, batch, logs=None):
                self.batch = self.batch + 1
                self.tbwriter.add_scalar(tag='train/batch_loss', scalar_value=logs['loss'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/batch_psnr', scalar_value=logs['psnr'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/batch_ssim', scalar_value=logs['ssim'], global_step=self.batch)

            def on_train_begin(self, logs=None):
                self.batch = 0
                self.tbwriter = SummaryWriter(root_path)

                mse_valid = imgProcess.mse(x_valid, y_valid)
                psnr_valid = imgProcess.psnr(x_valid, y_valid)
                ssim_valid = imgProcess.ssim(x_valid, y_valid)

                self.tbwriter.add_text(tag='note/initial', text_string=config_info, global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_loss', scalar_value=mse_valid, global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_psnr', scalar_value=psnr_valid.mean(), global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_ssim', scalar_value=ssim_valid.mean(), global_step=self.batch)

                for i in range(psnr_valid.shape[0]):
                    self.tbwriter.add_scalar(tag='valid_%d/epoch_psnr' % i, scalar_value=psnr_valid[i],
                                             global_step=self.batch)
                    self.tbwriter.add_scalar(tag='valid_%d/epoch_ssim' % i, scalar_value=ssim_valid[i],
                                             global_step=self.batch)
                    self.tbwriter.add_image(tag='valid_%d/x' % i,
                                            img_tensor=np.reshape(x_valid[i, :, :, :], [320, 320]),
                                            global_step=self.batch)

                    self.tbwriter.add_image(tag='valid_%d/y' % i,
                                            img_tensor=np.reshape(y_valid[i, :, :, :], [320, 320]),
                                            global_step=self.batch)

                sio.savemat(valid_path + 'init.mat', {
                    'x': x_valid,
                    'y': y_valid,
                })

                pass

        # Train Network
        self.net_train.fit(x=self.train_x, y=self.train_y, batch_size=self.batch_size, epochs=self.epochs,
                           callbacks=[ModelCheckpoint(filepath=self.model_path + '{epoch:02d}_weight.h5',
                                                      period=self.epoch_save, save_weights_only=True),
                                      SaveValidation(),
                                      ])
        self.net_train.save(self.model_path + 'final_model.h5')

    def build_network(self, shape_input, level=4, root_filters=64, rate_dropout=0.2, kernel_size=3):

        inputs = Input(shape=shape_input)
        conv = []

        for i in range(level):
            if i == 0:
                net = Conv2D(filters=root_filters * (2 ** i), kernel_size=kernel_size, padding='same')(inputs)
            else:
                net = Conv2D(filters=root_filters * (2 ** i), kernel_size=kernel_size, padding='same')(net)

            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = Dropout(rate_dropout)(net)

            net = Conv2D(filters=root_filters * (2 ** i), kernel_size=kernel_size, padding='same')(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = Dropout(rate_dropout)(net)

            conv.append(net)
            net = MaxPool2D(pool_size=(2, 2))(net)

        net = Conv2D(filters=root_filters * (2 ** level), kernel_size=kernel_size, padding='same')(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Dropout(rate_dropout)(net)

        net = Conv2D(filters=root_filters * (2 ** level), kernel_size=kernel_size, padding='same')(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Dropout(rate_dropout)(net)

        for i in range(level):
            net = Conv2DTranspose(filters=root_filters * (2 ** (level - 1 - i)), kernel_size=kernel_size, strides=2,
                                  padding='same')(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = Dropout(rate_dropout)(net)

            net = concatenate([net, conv[level - 1 - i]], axis=-1)

            net = Conv2D(filters=root_filters * (2 ** (level - 1 - i)), kernel_size=kernel_size, padding='same')(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = Dropout(rate_dropout)(net)

            net = Conv2D(filters=root_filters * (2 ** (level - 1 - i)), kernel_size=kernel_size, padding='same')(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = Dropout(rate_dropout)(net)

        net = Conv2D(filters=1, kernel_size=kernel_size, padding='same')(net)

        return keras.Model(inputs=inputs, outputs=net)

    def set_parameter(self):
        self.dataset_path = self.config['GLOBAL']['dataset_path']
        self.experiment_path = self.config['GLOBAL']['experiment_path']

        day = str(time.localtime().tm_mon) + str(time.localtime().tm_mday) + str(time.localtime().tm_year)
        method = 'unet'
        times = str(time.localtime().tm_hour) + str(time.localtime().tm_min)

        self.root_path = self.experiment_path + day + '/' + method + '_' + times + '/'
        self.valid_path = self.root_path + 'valid' + '/'
        self.model_path = self.root_path + 'model' + '/'

        # Training Parament
        self.batch_size = int(self.config['UNET']['batch_size'])
        self.epochs = int(self.config['UNET']['epochs'])
        self.epoch_save = int(self.config['UNET']['epoch_save'])
        self.learning_rate = float(self.config['UNET']['learning_rate'])

        # Make Sure the path is empty
        if os.path.exists(self.root_path):
            shutil.rmtree(self.root_path, ignore_errors=True)
        os.makedirs(self.root_path)

        if os.path.exists(self.valid_path):
            shutil.rmtree(self.valid_path, ignore_errors=True)
        os.makedirs(self.valid_path)

        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path, ignore_errors=True)
        os.makedirs(self.model_path)

        config_info = open('./config.ini', 'r').read()
        self.config_info = config_info

        print('Root Path: ' + self.root_path)
        print('Config Info' + self.config_info)

    def set_data(self):
        from util.dataLoader import AllUnetTrain, AllValid

        self.train_imgs = AllUnetTrain(self.dataset_path)
        self.valid_imgs = AllValid(self.dataset_path)

        self.train_x = self.train_imgs[0]
        self.train_y = self.train_imgs[1]

        self.valid_x = self.valid_imgs[0]
        self.valid_y = self.valid_imgs[1]

        self.shape_batch = (self.train_x[0].shape[0], self.train_x[0].shape[1], self.train_x[0].shape[2])
