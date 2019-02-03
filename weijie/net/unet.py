from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Dropout, \
    MaxPool2D, Conv2DTranspose, concatenate, Activation
from data import loader
from net.base import BaseKaresNetwork
import tensorflow as tf


class KerasNetwork(BaseKaresNetwork):

    def __init__(self):
        super().__init__(config_name='unet_')

    def get_loss(self):

        def old_total_variation_loss(y_true, y_pred):
            l2_loss = tf.losses.mean_squared_error(y_pred, y_true)
            tv_true_diff1 = y_true[:, 1:, :-1, :] - y_true[:, :-1, :-1, :]
            tv_true_diff2 = y_true[:, :-1, 1:, :] - y_true[:, :-1, :-1, :]
            tv_true = tf.abs(tv_true_diff1) + tf.abs(tv_true_diff2)

            tv_pre_diff1 = y_pred[:, 1:, :-1, :] - y_pred[:, :-1, :-1, :]
            tv_pre_diff2 = y_pred[:, :-1, 1:, :] - y_pred[:, :-1, :-1, :]
            tv_pre = tf.abs(tv_pre_diff1) + tf.abs(tv_pre_diff2)

            tv_loss = tf.losses.mean_squared_error(tv_pre, tv_true)
            return l2_loss + self.FLAGS_DICT[self.config_name + 'tv_lambda']*tv_loss

        def mse_loss(y_true, y_pred):
            return tf.losses.mean_squared_error(y_pred, y_true)

        def new_total_variation_loss(y_true, y_pred):
            l2_loss = tf.losses.mean_squared_error(y_pred, y_true)
            tv_true_diff1 = y_true[:, 1:, :-1, :] - y_true[:, :-1, :-1, :]
            tv_true_diff2 = y_true[:, :-1, 1:, :] - y_true[:, :-1, :-1, :]

            tv_pre_diff1 = y_pred[:, 1:, :-1, :] - y_pred[:, :-1, :-1, :]
            tv_pre_diff2 = y_pred[:, :-1, 1:, :] - y_pred[:, :-1, :-1, :]

            tv_diff1_loss = tf.losses.mean_squared_error(tv_true_diff1, tv_pre_diff1)
            tv_diff2_loss = tf.losses.mean_squared_error(tv_true_diff2, tv_pre_diff2)

            return l2_loss + self.FLAGS_DICT[self.config_name + 'tv_lambda']*(tv_diff1_loss + tv_diff2_loss)

        def total_variation_for_recononly_loss(y_true, y_pred):
            l2_loss = tf.losses.mean_squared_error(y_pred, y_true)
            tv_loss = tf.image.total_variation(y_pred)

            return l2_loss + self.FLAGS_DICT[self.config_name + 'tv_lambda']*tv_loss

        loss_types = {
            '0': mse_loss,
            '1': old_total_variation_loss,
            '2': new_total_variation_loss,
            '3': total_variation_for_recononly_loss,
        }

        return loss_types[self.FLAGS_DICT[self.config_name + 'loss_type']]

    def get_train_data(self):

        def source_mat_file():
            x_train, y_train, x_train_imgs, y_train_imgs = loader.mat2numpy(
                self.FLAGS_DICT['global_root_path'], self.FLAGS_DICT['global_dataset_type'],
                self.FLAGS_DICT[self.config_name + 'index_train_mat'],
                self.FLAGS_DICT[self.config_name + 'index_train_images'],
                self.FLAGS_DICT['global_debug'])

            x_val, y_val, x_val_imgs, y_val_imgs = loader.mat2numpy(
                self.FLAGS_DICT['global_root_path'], self.FLAGS_DICT['global_dataset_type'],
                self.FLAGS_DICT['global_index_valid_mat'],
                self.FLAGS_DICT['global_index_valid_images'],
                self.FLAGS_DICT['global_debug'])

            return x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs

        def old_liver_train_mat():
            x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs = \
                loader.load_old_unet_train(self.FLAGS_DICT['global_root_path'] +
                                           self.FLAGS_DICT['global_dataset_type'] + '/')

            return x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs

        data_types = {
            '0': source_mat_file,
            '1': old_liver_train_mat
        }

        return data_types[self.FLAGS_DICT[self.config_name + 'data_type']]()

    def get_network(self):

        input_shape = self.FLAGS_DICT[self.config_name + 'input_shape']
        level = self.FLAGS_DICT[self.config_name + 'level']
        root_filters = self.FLAGS_DICT[self.config_name + 'root_filters']
        rate_dropout = self.FLAGS_DICT[self.config_name + 'dropout']
        kernel_size = self.FLAGS_DICT[self.config_name + 'kernel_size']
        conv_time = self.FLAGS_DICT[self.config_name + 'conv_time']

        def conv_block(input_, fliters_coff):
            input_ = Conv2D(filters=root_filters * fliters_coff, kernel_size=kernel_size, padding='same')(input_)
            input_ = BatchNormalization()(input_)
            input_ = Activation('relu')(input_)
            input_ = Dropout(rate_dropout)(input_)
            return input_

        def dconv_block(input_, fliter_coff):
            input_ = Conv2DTranspose(filters=root_filters * fliter_coff, kernel_size=kernel_size, strides=2,
                                     padding='same')(input_)
            input_ = BatchNormalization()(input_)
            input_ = Activation('relu')(input_)
            input_ = Dropout(rate_dropout)(input_)
            return input_

        inputs_ = Input(shape=input_shape)
        net = conv_block(inputs_, 1)
        conv = []

        for i in range(level):
            for j in range(conv_time):
                net = conv_block(net, 2 ** i)

            conv.append(net)
            net = MaxPool2D(pool_size=(2, 2))(net)

        for j in range(conv_time):
            net = conv_block(net, 2 ** level)

        for i in range(level):
            net = dconv_block(net, 2 ** (level - 1 - i))
            net = concatenate([net, conv[level - 1 - i]], axis=-1)
            for j in range(conv_time):
                net = conv_block(net, 2 ** (level - 1 - i))

        net = Conv2D(filters=1, kernel_size=1, padding='same')(net)

        return Model(inputs=inputs_, outputs=net)
