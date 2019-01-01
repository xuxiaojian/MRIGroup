import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, \
    MaxPool2D, concatenate, Activation, UpSampling2D
from tensorflow.keras.optimizers import Adam
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorboardX import SummaryWriter
from util import imgProcess
import numpy as np


class KerasNetwork(object):

    def __init__(self, config, shape_input):

        self.set_parameter(config)
        self.shape_input = shape_input

        self.net_train = self.build_network()
        self.net_train.summary()
        self.net_train = keras.utils.multi_gpu_model(self.net_train, gpus=2)

    def train(self, x_train, y_train, x_val, y_val):

        # Complie Network
        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, max_val=1)

        def ssim(y_true, y_pred):
            return tf.image.ssim(y_pred, y_true, max_val=1)

        self.net_train.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr, ssim])

        # Make Sure the Svae Path is empty
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path, ignore_errors=True)
        os.makedirs(self.save_path)

        config_info = self.config_info
        save_path = self.save_path

        class SaveValidation(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.tbwriter.add_scalar(tag='train/epoch_loss', scalar_value=logs['loss'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/epoch_psnr', scalar_value=logs['psnr'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/epoch_ssim', scalar_value=logs['ssim'], global_step=self.batch)

                pre_valid = self.model.predict(x_val)
                mse_valid = imgProcess.mse(pre_valid, y_val)
                psnr_valid = imgProcess.psnr(pre_valid, y_val)
                ssim_valid = imgProcess.ssim(pre_valid, y_val)

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
                    pre_valid_ = imgProcess.normalize(pre_valid_)

                    self.tbwriter.add_image(tag='valid_%d/predict' % i,
                                            img_tensor=np.reshape(pre_valid_, [320, 320]),
                                            global_step=self.batch)

            def on_batch_end(self, batch, logs=None):
                self.batch = self.batch + 1
                self.tbwriter.add_scalar(tag='train/batch_loss', scalar_value=logs['loss'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/batch_psnr', scalar_value=logs['psnr'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/batch_ssim', scalar_value=logs['ssim'], global_step=self.batch)

            def on_train_begin(self, logs=None):
                self.batch = 0
                self.tbwriter = SummaryWriter(save_path)

                mse_valid = imgProcess.mse(x_val, y_val)
                psnr_valid = imgProcess.psnr(x_val, y_val)
                ssim_valid = imgProcess.ssim(x_val, y_val)

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
                                            img_tensor=np.reshape(x_val[i, :, :, :], [320, 320]),
                                            global_step=self.batch)

                    self.tbwriter.add_image(tag='valid_%d/y' % i,
                                            img_tensor=np.reshape(y_val[i, :, :, :], [320, 320]),
                                            global_step=self.batch)

        # Train Network
        self.net_train.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs,
                           callbacks=[ModelCheckpoint(filepath=self.save_path + '{epoch:02d}_weight.h5',
                                                      period=50, save_weights_only=True),
                                      SaveValidation(),
                                      ])
        self.net_train.save_weights(self.save_path + 'final_weight.h5')

    def build_network(self):

        inputs = Input(shape=self.shape_input)
        conv = []

        for i in range(self.net_level):
            if i == 0:
                net = Conv2D(filters=self.root_filters * (2 ** i), kernel_size=self.kernel_size, padding='same')(inputs)
            else:
                net = Conv2D(filters=self.root_filters * (2 ** i), kernel_size=self.kernel_size, padding='same')(net)

            net = Activation('relu')(net)
            net = BatchNormalization()(net)
            net = Dropout(self.rate_dropout)(net)

            net = Conv2D(filters=self.root_filters * (2 ** i), kernel_size=self.kernel_size, padding='same')(net)
            net = Activation('relu')(net)
            net = BatchNormalization()(net)
            net = Dropout(self.rate_dropout)(net)

            conv.append(net)
            net = MaxPool2D(pool_size=(2, 2))(net)

        net = Conv2D(filters=self.root_filters * (2 ** self.net_level), kernel_size=self.kernel_size, padding='same')(net)
        net = Activation('relu')(net)
        net = BatchNormalization()(net)
        net = Dropout(self.rate_dropout)(net)

        net = Conv2D(filters=self.root_filters * (2 ** self.net_level), kernel_size=self.kernel_size, padding='same')(net)
        net = Activation('relu')(net)
        net = BatchNormalization()(net)
        net = Dropout(self.rate_dropout)(net)

        for i in range(self.net_level):
            # net = Conv2DTranspose(filters=self.root_filters * (2 ** (self.net_level - 1 - i)),
            #                       kernel_size=self.kernel_size, strides=2,
            #                       padding='same')(net)
            net = UpSampling2D()(net)
            net = Activation('relu')(net)
            net = BatchNormalization()(net)
            net = Dropout(self.rate_dropout)(net)

            net = concatenate([net, conv[self.net_level - 1 - i]], axis=-1)

            net = Conv2D(filters=self.root_filters * (2 ** (self.net_level - 1 - i)),
                         kernel_size=self.kernel_size, padding='same')(net)
            net = Activation('relu')(net)
            net = BatchNormalization()(net)
            net = Dropout(self.rate_dropout)(net)

            net = Conv2D(filters=self.root_filters * (2 ** (self.net_level - 1 - i)),
                         kernel_size=self.kernel_size, padding='same')(net)
            net = Activation('relu')(net)
            net = BatchNormalization()(net)
            net = Dropout(self.rate_dropout)(net)

        net = Conv2D(filters=1, kernel_size=self.kernel_size, padding='same')(net)

        return keras.Model(inputs=inputs, outputs=net)

    def set_parameter(self, config):
        self.experiment_path = config['GLOBAL']['experiment_path']

        self.save_path = self.experiment_path + config['UNET']['save_path']

        # Training Parament
        self.batch_size = int(config['UNET']['batch_size'])
        self.epochs = int(config['UNET']['epochs'])
        self.learning_rate = float(config['UNET']['learning_rate'])

        # Network Parameter
        self.net_level = int(config['UNET']['net_level'])
        self.root_filters = int(config['UNET']['root_filters'])
        self.rate_dropout = float(config['UNET']['rate_dropout'])
        self.kernel_size = int(config['UNET']['kernel_size'])

        config_file = open('./config.ini', 'r')
        self.config_info = config_file.read()
        config_file.close()

        print('Model Path: ' + self.save_path)
        print('Config Info' + self.config_info)
