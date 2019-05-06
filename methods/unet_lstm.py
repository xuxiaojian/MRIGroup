from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, Input, MaxPool3D, \
    Concatenate, Add, TimeDistributed, Lambda, RNN
from tensorflow.python.keras.models import Model
import numpy as np
import tensorflow as tf


class Conv3DLSTM(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(Conv3DLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.rnn_cell = tf.contrib.rnn.Conv3DLSTMCell(
            input_shape=[int(input_shape[-4]), int(input_shape[-3]), int(input_shape[-2]), int(input_shape[-1])],
            output_channels=self.filters,
            kernel_shape=[self.kernel_size, self.kernel_size, self.kernel_size],
            skip_connection=False
        )
        # noinspection PyAttributeOutsideInit
        self.rnn = RNN(self.rnn_cell, return_sequences=True)

        super(Conv3DLSTM, self).build(input_shape)

    def call(self, x, **kwargs):

        return self.rnn(x)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return input_shape


def unet_3d_lstm(config, key):

    input_shape = np.fromstring(config[key]['input_shape'], dtype=np.int, sep=',').tolist()
    kernel_size = int(config[key]['kernel_size'])
    filters_root = int(config[key]['filters_root'])
    conv_times = int(config[key]['conv_times'])
    up_down_times = int(config[key]['up_down_times'])

    # def conv3d_relu_dropout_parallel_with_lstm(input_, filters_, kernel_size_):
    #     output_ = TimeDistributed(Conv3D(filters=filters_, kernel_size=kernel_size_, padding='same'))(input_)
    #     output_ = TimeDistributed(ReLU())(output_)
    #     output_ = TimeDistributed(Dropout(rate=0.1))(output_)
    #     return output_
    #
    # rnn_cell = tf.contrib.rnn.Conv3DLSTMCell(
    #     input_shape=[10, 320, 320, 1],
    #     output_channels=32,
    #     kernel_shape=[3, 3, 3],
    #     skip_connection=False
    # )
    #
    net_input = Input(shape=input_shape)
    net = Conv3DLSTM(32, 3)(net_input)
    model = Model(inputs=net_input, outputs=net)

    return model
