from net.base import BaseKaresNetwork, psnr_metrics, ssim_metrics
from data import loader
from tensorflow import keras as keras
import numpy as np
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from data.recorder import normalize


class KerasNetwork(BaseKaresNetwork):

    def __init__(self):
        super().__init__(config_name='sr_')

    def get_loss(self):
        return keras.losses.mean_absolute_error

    def get_network(self):
        input_shape = self.FLAGS_DICT[self.config_name + 'input_shape']
        filters = self.FLAGS_DICT[self.config_name + 'filters']
        num_rblocks = self.FLAGS_DICT[self.config_name + 'num_rblocks']
        kernel_size_rblocks = self.FLAGS_DICT[self.config_name + 'kernel_size_rblocks']
        kernel_size_io = self.FLAGS_DICT[self.config_name + 'kernel_size_io']

        input_data = Input(shape=input_shape)
        net = Conv2D(filters=filters, kernel_size=kernel_size_io, padding='same', name='conv2d_input')(input_data)
        input_net = net

        for i in range(num_rblocks):
            input_rblock = net
            #########################
            # BLOCK DEFINITION
            #########################
            net = Conv2D(filters=filters, kernel_size=kernel_size_rblocks, padding='same', name='conv2d1_nbl%d' % i)(net)
            net = BatchNormalization(name='bn1_nbl%d' % i)(net)

            net = Activation('relu', name='relu_nbl%d' % i)(net)

            net = Conv2D(filters=filters, kernel_size=kernel_size_rblocks, padding='same', name='conv2d2_nbl%d' % i)(net)
            net = BatchNormalization(name='bn2_nbl%d' % i)(net)

            net = Add(name='add_nbl%d' % i)([net, input_rblock])
            #########################
            #########################

        net = Conv2D(filters=filters, kernel_size=kernel_size_rblocks, padding='same', name='conv2d_output1')(net)
        net = Add(name='add_output1')([net, input_net])

        net = Conv2D(filters=filters, kernel_size=kernel_size_rblocks, padding='same', name='conv2d_output2')(net)
        net = Conv2D(filters=1, kernel_size=kernel_size_io, padding='same', name='final_output')(net)

        return keras.Model(inputs=input_data, outputs=net)

    def get_train_data(self):

        def source_mat_file():
            x_train_feature1, y_train, x_train_imgs_feature1, y_train_imgs = loader.mat2numpy(
                self.FLAGS_DICT['global_root_path'], self.FLAGS_DICT['global_dataset_type'],
                self.FLAGS_DICT[self.config_name + 'index_train_mat'],
                self.FLAGS_DICT[self.config_name + 'index_train_images'],
                self.FLAGS_DICT['global_debug'])

            x_val_feature1, y_val, x_val_imgs_feature1, y_val_imgs = loader.mat2numpy(
                self.FLAGS_DICT['global_root_path'], self.FLAGS_DICT['global_dataset_type'],
                self.FLAGS_DICT['global_index_valid_mat'],
                self.FLAGS_DICT['global_index_valid_images'],
                self.FLAGS_DICT['global_debug'])

            from net import dearti
            unet_model = dearti.KerasNetwork()
            unet_model.network.load_weights(self.FLAGS_DICT[self.config_name + 'unet_model_path'])

            x_train_feature2 = unet_model.network.predict(x_train_feature1, verbose=1)
            x_train_imgs_feature2 = unet_model.network.predict(x_train_imgs_feature1, verbose=1)
            x_val_feature2 = unet_model.network.predict(x_val_feature1, verbose=1)
            x_val_imgs_feature2 = unet_model.network.predict(x_val_imgs_feature1, verbose=1)

            batch_size = x_train_feature2.shape[0]
            for i in range(batch_size):
                x_train_feature2[i, :, :, :] = normalize(x_train_feature2[i, :, :, :])

            batch_size = x_train_imgs_feature2.shape[0]
            for i in range(batch_size):
                x_train_imgs_feature2[i, :, :, :] = normalize(x_train_imgs_feature2[i, :, :, :])

            batch_size = x_val_feature2.shape[0]
            for i in range(batch_size):
                x_val_feature2[i, :, :, :] = normalize(x_val_feature2[i, :, :, :])

            batch_size = x_val_imgs_feature2.shape[0]
            for i in range(batch_size):
                x_val_imgs_feature2[i, :, :, :] = normalize(x_val_imgs_feature2[i, :, :, :])

            return x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
                y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, y_val_imgs

        def old_liver_train_mat():
            x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2,\
                y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, \
                y_val_imgs = loader.load_old_sr_train(self.FLAGS_DICT['global_root_path']
                                                      + self.FLAGS_DICT['global_dataset_type'] + '/')

            return x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2,\
                y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, y_val_imgs

        def xiaojian_liver_train_mat():
            x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
                y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, \
                y_val_imgs = loader.load_xiaojian_sr_train(self.FLAGS_DICT['global_root_path'] +
                                                           self.FLAGS_DICT['global_dataset_type'] + '/')

            return x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
                y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, y_val_imgs

        def load_xiaojian_sr_train_feb14():
            x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
            y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, \
            y_val_imgs = loader.load_xiaojian_sr_train_feb14(self.FLAGS_DICT['global_root_path'] +
                                                       self.FLAGS_DICT['global_dataset_type'] + '/')

            return x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
                   y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, y_val_imgs

        data_types = {
            '0': source_mat_file,
            '1': old_liver_train_mat,
            '2': xiaojian_liver_train_mat,
            '3': load_xiaojian_sr_train_feb14,
        }

        x_train_feature1, x_train_feature2, y_train, x_train_imgs_feature1, x_train_imgs_feature2, \
            y_train_imgs, x_val_feature1, x_val_feature2, y_val, x_val_imgs_feature1, x_val_imgs_feature2, y_val_imgs = \
            data_types[self.FLAGS_DICT[self.config_name + 'data_type']]()

        x_train = np.concatenate([x_train_feature1, x_train_feature2], axis=-1)
        x_train_imgs = np.concatenate([x_train_imgs_feature1, x_train_imgs_feature2], axis=-1)
        x_val = np.concatenate([x_val_feature1, x_val_feature2], axis=-1)
        x_val_imgs = np.concatenate([x_val_imgs_feature1, x_val_imgs_feature2], axis=-1)

        x_train = loader.crop_images(x_train,
                                     self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                     self.FLAGS_DICT[self.config_name + 'crop_step'])
        y_train = loader.crop_images(y_train,
                                     self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                     self.FLAGS_DICT[self.config_name + 'crop_step'])

        x_val = loader.crop_images(x_val,
                                   self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                   self.FLAGS_DICT[self.config_name + 'crop_step'])
        y_val = loader.crop_images(y_val,
                                   self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                   self.FLAGS_DICT[self.config_name + 'crop_step'])

        return x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs
