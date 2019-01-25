from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Dropout, \
    MaxPool2D, Conv2DTranspose, concatenate, Activation
from util import data_loader
from net.base import BaseKaresNetwork


class KerasNetwork(BaseKaresNetwork):

    def __init__(self):
        super().__init__(config_name='unet_')

    def set_train_imgs(self):
        x_train, y_train, x_train_imgs, y_train_imgs = data_loader.mat2numpy(
            self.FLAGS_DICT['root_path'], self.FLAGS_DICT['dataset_type'],
            self.FLAGS_DICT[self.config_name + 'index_train_mat'],
            self.FLAGS_DICT[self.config_name + 'index_train_images'])

        x_val, y_val, x_val_imgs, y_val_imgs = data_loader.mat2numpy(
            self.FLAGS_DICT['root_path'], self.FLAGS_DICT['dataset_type'],
            self.FLAGS_DICT['index_valid_mat'],
            self.FLAGS_DICT['index_valid_images'])

        return x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs

    def set_network(self):

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

        net = Conv2D(filters=1, kernel_size=kernel_size, padding='same')(net)

        return Model(inputs=inputs_, outputs=net)
