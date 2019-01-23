from tensorflow import keras as keras
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Dropout, \
    MaxPool2D, Conv2DTranspose, concatenate, Activation
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
from util.data_recorder import psnr_metrics, ssim_metrics, KearsCallback, new_folder
from tensorflow.python.keras.callbacks import ModelCheckpoint
from util import data_loader


class KerasNetwork(object):

    def __init__(self):
        self.FLAGS = tf.flags.FLAGS
        self.network = self.build_network()
        self.network.summary()
        self.network.compile(optimizer=Adam(lr=self.FLAGS.unet_learning_rate),
                             loss=keras.losses.mean_squared_error,
                             metrics=[psnr_metrics, ssim_metrics])

    def __call__(self, mode):
        if mode == 'train':
            self.train()

    def set_train_images(self):
        self.data_train, self.label_train, self.imgs_data_train, self.imgs_label_train = data_loader.mat2numpy(
            self.FLAGS.root_path, self.FLAGS.dataset_type,
            self.FLAGS.index_train_unet_mat, self.FLAGS.index_train_unet_images)

        self.data_valid, self.label_valid, self.imgs_data_valid, self.imgs_label_valid = data_loader.mat2numpy(
            self.FLAGS.root_path, self.FLAGS.dataset_type,
            self.FLAGS.index_valid_mat, self.FLAGS.index_valid_images)

    def train(self):
        self.set_train_images()

        model_path = self.FLAGS.unet_output_path + 'model/'
        new_folder(self.FLAGS.unet_output_path)
        new_folder(model_path)

        costom_callback = KearsCallback(self.imgs_data_train, self.imgs_label_train,
                                        self.imgs_data_valid, self.imgs_label_valid,
                                        self.FLAGS.unet_output_path, self.FLAGS.flags_into_string())
        save_model_callback = ModelCheckpoint(filepath=model_path + '{epoch:02d}_weight.h5',
                                              period=self.FLAGS.unet_epoch_save_model, save_weights_only=True)

        self.network.fit(x=self.data_train, y=self.label_train,
                         batch_size=self.FLAGS.unet_batch_size,
                         epochs=self.FLAGS.unet_epoch,
                         validation_data=(self.data_valid, self.label_valid),
                         callbacks=[costom_callback, save_model_callback])

        self.network.save_weights(model_path + 'final_weight.h5')
        self.network.save(model_path + 'final_model.h5')

        pass

    def build_network(self):

        level = self.FLAGS.unet_net_level
        root_filters = self.FLAGS.unet_net_root_filters
        rate_dropout = self.FLAGS.unet_net_dropout
        kernel_size = self.FLAGS.unet_net_kernel_size

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

        inputs_ = Input(shape=self.FLAGS.unet_net_input_shape)
        conv = []
        net = None

        for i in range(level):
            if i == 0:
                net = conv_block(inputs_, 2 ** i)
            else:
                net = conv_block(net, 2 ** i)

            net = conv_block(net, 2 ** i)
            conv.append(net)
            net = MaxPool2D(pool_size=(2, 2))(net)

        net = conv_block(net, 2 ** level)
        net = conv_block(net, 2 ** level)

        for i in range(level):
            net = dconv_block(net, 2 ** (level - 1 - i))
            net = concatenate([net, conv[level - 1 - i]], axis=-1)
            net = conv_block(net, 2 ** (level - 1 - i))
            net = conv_block(net, 2 ** (level - 1 - i))

        net = Conv2D(filters=1, kernel_size=kernel_size, padding='same')(net)

        return keras.Model(inputs=inputs_, outputs=net)
