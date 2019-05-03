from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, Input, MaxPool3D, \
    Concatenate, Add
from tensorflow.python.keras.models import Model
import numpy as np


def unet_3d(config, key):
    input_shape = np.fromstring(config[key]['input_shape'], dtype=np.int, sep=',').tolist()
    kernel_size = int(config[key]['kernel_size'])
    filters_root = int(config[key]['filters_root'])
    conv_times = int(config[key]['conv_times'])
    up_down_times = int(config[key]['up_down_times'])

    def conv3d_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv3D(filters=filters_, kernel_size=kernel_size_, padding='same')(input_)
        output_ = ReLU()(output_)
        output_ = Dropout(rate=0.1)(output_)
        return output_

    def conv3d_transpose_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(1, 2, 2))(input_)
        output_ = ReLU()(output_)
        output_ = Dropout(rate=0.1)(output_)
        return output_

    skip_layers_storage = []

    # Build Network
    net_input = Input(shape=input_shape)
    net = conv3d_relu_dropout(net_input, filters_root, kernel_size)

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size)

        skip_layers_storage.append(net)
        net = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(net)

    for layer in range(up_down_times - 1, -1, -1):

        filters = 2 ** layer * filters_root
        net = conv3d_transpose_relu_dropout(net, filters, kernel_size)
        net = Concatenate(axis=-1)([net, skip_layers_storage[layer]])

        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size)

    net = Conv3D(filters=1, kernel_size=1, padding='same')(net)
    net = Add()([net, net_input])

    model = Model(inputs=net_input, outputs=net)

    return model
