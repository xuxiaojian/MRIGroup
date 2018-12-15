import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import shutil
import tensorflow as tf
import scipy.io as sio
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorboardX import SummaryWriter
from util import imgProcess
import numpy as np


class KerasNetwork(object):

    def __init__(self, config, dataset_path):

        self.config = config
        self.dataset_path = dataset_path

        self.set_parameter(config)
        self.set_data(config)

        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, max_val=1)

        def ssim(y_true, y_pred):
            return tf.image.ssim(y_pred, y_true, max_val=1)

        self.net_train = self.build_network(self.shape_train)
        self.net_train.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr, ssim])
        self.net_train.summary()

        self.net_valid = self.build_network(self.shape_valid)
        self.net_valid.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr, ssim])

    def train(self):

        note = self.note
        net_valid = self.net_valid
        x_noised_valid, x_valid, y_valid = self.valid_imgs
        root_path = self.root_path

        epoch_save = self.epoch_save
        valid_path = self.valid_path

        class SaveValidation(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.tbwriter.add_scalar(tag='train/epoch_loss', scalar_value=logs['loss'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/epoch_psnr', scalar_value=logs['psnr'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/epoch_ssim', scalar_value=logs['ssim'], global_step=self.batch)

                net_valid.set_weights(self.model.get_weights())
                pre_valid = net_valid.predict(x_valid)
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
                    self.tbwriter.add_image(tag='valid_%d/predict' % i,
                                            img_tensor=np.reshape(pre_valid[i, :, :, :], [320, 320]),
                                            global_step=self.batch)

            def on_batch_end(self, batch, logs=None):
                self.batch = self.batch + 1
                self.tbwriter.add_scalar(tag='train/batch_loss', scalar_value=logs['loss'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/batch_psnr', scalar_value=logs['psnr'], global_step=self.batch)
                self.tbwriter.add_scalar(tag='train/batch_ssim', scalar_value=logs['ssim'], global_step=self.batch)

            def on_train_begin(self, logs=None):
                self.batch = 0
                self.tbwriter = SummaryWriter(root_path)

                mse_valid = imgProcess.mse(x_noised_valid, y_valid)
                psnr_valid = imgProcess.psnr(x_noised_valid, y_valid)
                ssim_valid = imgProcess.ssim(x_noised_valid, y_valid)

                self.tbwriter.add_text(tag='note/initial', text_string=note, global_step=self.batch)
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

                    self.tbwriter.add_image(tag='valid_%d/x_noised' % i,
                                            img_tensor=np.reshape(x_noised_valid[i, :, :, :], [320, 320]),
                                            global_step=self.batch)

                pass

        # Train Network
        self.net_train.fit(x=self.train_imgs[1], y=self.train_imgs[2], batch_size=self.batch_size, epochs=self.epochs,
                           callbacks=[ModelCheckpoint(filepath=self.model_path + '{epoch:02d}_weight.h5',
                                                      period=self.epoch_save, save_weights_only=True),
                                      SaveValidation(),
                                      ])
        self.net_train.save(self.model_path + 'final_model.h5')

    def build_network(self, shape_input, num_rblocks=16, num_filters=64):

        input_data = Input(shape=shape_input)
        net = Conv2D(filters=num_filters, kernel_size=9, padding='same', name='conv2d_input')(input_data)
        # kernel_size = 9 is learnt from SRResNet
        input_net = net

        for i in range(num_rblocks):
            input_rblock = net
            net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d1_nbl%d' % i)(net)
            net = BatchNormalization(name='bn1_nbl%d' % i)(net)

            net = Activation('relu', name='relu_nbl%d' % i)(net)

            net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d2_nbl%d' % i)(net)
            net = BatchNormalization(name='bn2_nbl%d' % i)(net)

            net = Add(name='add_nbl%d' % i)([net, input_rblock])

        net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d_output1')(net)
        net = Add(name='add_output1')([net, input_net])
        net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d_output2')(net)
        net = Conv2D(filters=1, kernel_size=9, padding='same', name='final_output')(net)

        return keras.Model(inputs=input_data, outputs=net)

    def set_parameter(self, config):
        self.note = self.config['SR']['note']

        # Saving Parameter
        self.model_path = self.config['SR']['model_path']
        self.valid_path = self.config['SR']['valid_path']
        self.root_path = self.config['SR']['root_path']

        # Training Parament
        self.batch_size = int(self.config['SR']['batch_size'])
        self.epochs = int(self.config['SR']['epochs'])
        self.epoch_save = int(self.config['SR']['epoch_save'])
        self.learning_rate = float(self.config['SR']['learning_rate'])

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

        shutil.copy('./config.ini', self.root_path)

    def set_data(self, config):
        from util.dataLoader import LiverSRTrain, LiverValid

        self.patch_size = int(config['SR']['patch_size'])
        self.patch_step = int(config['SR']['patch_step'])

        self.train_imgs = LiverSRTrain(self.dataset_path, cropped=True, size=(self.patch_size, self.patch_size),
                                       step=self.patch_step)
        self.valid_imgs = LiverValid(self.dataset_path)

        self.shape_train = (self.train_imgs[0].shape[1], self.train_imgs[0].shape[2], self.train_imgs[0].shape[3])
        self.shape_valid = (self.valid_imgs[0].shape[1], self.valid_imgs[0].shape[2], self.valid_imgs[0].shape[3])
