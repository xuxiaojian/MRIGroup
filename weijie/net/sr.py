from net.base import BaseKaresNetwork
from util import data_loader
from tensorflow import keras as keras
import numpy as np
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from net import unet


class KerasNetwork(BaseKaresNetwork):

    def __init__(self):
        super().__init__(config_name='sr_')

    def set_network(self):
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

    def set_train_imgs(self):

        x_train_feature1, y_train, x_train_imgs_feature1, y_train_imgs = data_loader.mat2numpy(
            self.FLAGS_DICT['root_path'], self.FLAGS_DICT['dataset_type'],
            self.FLAGS_DICT[self.config_name + 'index_train_mat'],
            self.FLAGS_DICT[self.config_name + 'index_train_images'],
            self.FLAGS_DICT['debug'])

        x_val__feature1, y_val, x_val_imgs_feature1, y_val_imgs = data_loader.mat2numpy(
            self.FLAGS_DICT['root_path'], self.FLAGS_DICT['dataset_type'],
            self.FLAGS_DICT['index_valid_mat'],
            self.FLAGS_DICT['index_valid_images'],
            self.FLAGS_DICT['debug'])

        unet_model = unet.KerasNetwork()
        unet_model.network.load_weights(self.FLAGS_DICT[self.config_name + 'unet_model_path'])
        x_train_feature2 = unet_model.network.predict(x_train_feature1, verbose=1)
        x_train_imgs_feature2 = unet_model.network.predict(x_train_imgs_feature1, verbose=1)
        x_val_feature2 = unet_model.network.predict(x_val__feature1, verbose=1)
        x_val_imgs_feature2 = unet_model.network.predict(x_val_imgs_feature1, verbose=1)

        x_train = np.concatenate([x_train_feature1, x_train_feature2], axis=-1)
        x_train_imgs = np.concatenate([x_train_imgs_feature1, x_train_imgs_feature2], axis=-1)
        x_val = np.concatenate([x_val__feature1, x_val_feature2], axis=-1)
        x_val_imgs = np.concatenate([x_val_imgs_feature1, x_val_imgs_feature2], axis=-1)

        x_train = data_loader.crop_images(x_train,
                                          self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                          self.FLAGS_DICT[self.config_name + 'crop_step'])
        y_train = data_loader.crop_images(y_train,
                                          self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                          self.FLAGS_DICT[self.config_name + 'crop_step'])

        x_val = data_loader.crop_images(x_val,
                                        self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                        self.FLAGS_DICT[self.config_name + 'crop_step'])
        y_val = data_loader.crop_images(y_val,
                                        self.FLAGS_DICT[self.config_name + 'crop_window_size'],
                                        self.FLAGS_DICT[self.config_name + 'crop_step'])

        return x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs

    def set_test_imgs(self):
        pass
