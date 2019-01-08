import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorboardX import SummaryWriter
from util import imgProcess
import numpy as np


class KerasNetwork(object):

    def __init__(self, config, channel,  num_gpu=1):

        self.set_parameter(config)
        self.num_gpu = num_gpu

        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, max_val=1)

        def ssim(y_true, y_pred):
            return tf.image.ssim(y_pred, y_true, max_val=1)

        self.net_train = self.build_network([self.patch_size, self.patch_size, channel])
        if num_gpu > 1:
            self.net_train = keras.utils.multi_gpu_model(self.net_train, num_gpu)
        self.net_train.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr, ssim])
        self.net_train.summary()

        self.net_valid = self.build_network([320, 320, channel])
        if num_gpu > 1:
            self.net_valid = keras.utils.multi_gpu_model(self.net_valid, num_gpu)
        self.net_valid.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr, ssim])

    def train(self, x_train, y_train, x_valid, y_valid):

        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path, ignore_errors=True)
        os.makedirs(self.save_path)

        x_train, y_train = self.__crop_data(x_train, y_train, self.patch_size, self.patch_step)

        config_info = self.config_info
        net_valid = self.net_valid
        save_path = self.save_path
        batch_save = self.batch_save

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
                if batch % batch_save == 0:
                    self.batch = self.batch + 1
                    self.tbwriter.add_scalar(tag='train/batch_loss', scalar_value=logs['loss'], global_step=self.batch)
                    self.tbwriter.add_scalar(tag='train/batch_psnr', scalar_value=logs['psnr'], global_step=self.batch)
                    self.tbwriter.add_scalar(tag='train/batch_ssim', scalar_value=logs['ssim'], global_step=self.batch)

            def on_train_begin(self, logs=None):
                self.batch = 0
                self.tbwriter = SummaryWriter(save_path)

                x_valid_noised = x_valid[:, :, :, 0]
                x_valid_noised.shape = y_valid.shape

                mse_valid = imgProcess.mse(x_valid_noised, y_valid)
                psnr_valid = imgProcess.psnr(x_valid_noised, y_valid)
                ssim_valid = imgProcess.ssim(x_valid_noised, y_valid)

                self.tbwriter.add_text(tag='note/initial', text_string=config_info, global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_loss', scalar_value=mse_valid, global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_psnr', scalar_value=psnr_valid.mean(), global_step=self.batch)
                self.tbwriter.add_scalar(tag='valid_avg/epoch_ssim', scalar_value=ssim_valid.mean(), global_step=self.batch)

                for i in range(psnr_valid.shape[0]):
                    self.tbwriter.add_scalar(tag='valid_%d/epoch_psnr' % i, scalar_value=psnr_valid[i],
                                             global_step=self.batch)
                    self.tbwriter.add_scalar(tag='valid_%d/epoch_ssim' % i, scalar_value=ssim_valid[i],
                                             global_step=self.batch)
                    self.tbwriter.add_image(tag='valid_%d/y' % i,
                                            img_tensor=np.reshape(y_valid[i, :, :, :], [320, 320]),
                                            global_step=self.batch)

                    for j in range(x_valid.shape[3]):

                        self.tbwriter.add_image(tag='valid_%d/x_%d' % (i, j),
                                                img_tensor=np.reshape(x_valid[i, :, :, j], [320, 320]),
                                                global_step=self.batch)

                pass

        # gpu_config = tf.ConfigProto()
        # gpu_config.gpu_options.allow_growth = True
        # gpu_config.allow_soft_placement = True
        # sess = tf.Session(config=gpu_config)
        # keras.backend.set_session(session=sess)

        # Train Network
        self.net_train.fit(x=x_train, y=y_train, batch_size=self.batch_size * self.num_gpu, epochs=self.epochs,
                           callbacks=[ModelCheckpoint(filepath=self.save_path + '{epoch:02d}_weight.h5',
                                                      period=self.epoch_save, save_weights_only=True),
                                      SaveValidation(),
                                      ])
        self.net_train.save(self.save_path + 'final_model.h5')

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

        self.save_path = config['GLOBAL']['save_path']

        self.patch_size = int(config['SR']['patch_size'])
        self.patch_step = int(config['SR']['patch_step'])

        self.batch_size = int(config['SR']['batch_size'])
        self.epochs = int(config['SR']['epochs'])
        self.epoch_save = int(config['SR']['epoch_save'])
        self.batch_save = int(config['SR']['batch_save'])
        self.learning_rate = float(config['SR']['learning_rate'])

        config_info = open('./config.ini', 'r')
        self.config_info = config_info.read()
        config_info.close()

    def __crop_data(self, x, y, patch_size, patch_step):

        from skimage.util import view_as_windows
        print('Begin Cropping Train Images....')

        batchsize = x.shape[0]
        channel_x = x.shape[3]
        channel_y = y.shape[3]

        x_crops = view_as_windows(x, [batchsize, patch_size, patch_size, channel_x], patch_step)
        y_crops = view_as_windows(y, [batchsize, patch_size, patch_size, channel_y], patch_step)

        num_patch = x_crops.shape[1] * x_crops.shape[2] * batchsize
        print('Cropping Done, Total [%d] Patches' % num_patch)

        print('Begin Reshape Array....')
        x_crops = np.ascontiguousarray(x_crops)
        y_crops = np.ascontiguousarray(y_crops)
        x_crops.shape = [num_patch, patch_size, patch_size, channel_x]
        y_crops.shape = [num_patch, patch_size, patch_size, channel_y]
        print('Reshape Array Done.')

        # from skimage.io import imsave
        # imsave('noised.png', x_crops[100, :, :, 0])
        # imsave('feature.png', x_crops[100, :, :, 1])
        # imsave('clear.png', y_crops[100, :, :, 0])

        # exit(233)
        return x_crops, y_crops
