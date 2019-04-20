from method.tfbase import TFBase
import tensorflow as tf
import numpy as np
import torch
from torch import nn


class TorchNet3D(torch.nn.Module):
    def __init__(self, config):
        super(TorchNet3D, self).__init__()

        # Network Parameters
        self.config = config
        self.conv_ks = int(self.config['torch-unet3d']['conv_ks'])
        padding = int((self.conv_ks - 1) / 2)

        self.conv_filter_root = int(self.config['torch-unet3d']['conv_filter_root'])
        self.conv_times = int(self.config['torch-unet3d']['conv_times'])

        ##### Trainable #####

        # 1st
        filters = 2 ** 0 * self.conv_filter_root
        self.Conv3d1DownIn = nn.Conv3d(1, filters, self.conv_ks, padding=padding)
        self.Conv3d1DownList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        self.Conv3dTranspose1 = nn.ConvTranspose3d(2 ** 1 * self.conv_filter_root, filters, self.conv_ks, padding=padding, output_padding=[0, 1, 1], stride=[1, 2, 2])
        self.Conv3d1UpIn = nn.Conv3d(filters*2, filters, self.conv_ks, padding=padding)
        self.Conv3d1UpList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        self.Conv3dOutput = nn.Conv3d(filters, 1, self.conv_ks, padding=padding)

        # 2nd
        filters = 2 ** 1 * self.conv_filter_root
        self.Conv3d2DownIn = nn.Conv3d(2 ** 0 * self.conv_filter_root, filters, self.conv_ks, padding=padding)
        self.Conv3d2DownList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        self.Conv3dTranspose2 = nn.ConvTranspose3d(2 ** 2 * self.conv_filter_root, filters, self.conv_ks,padding=padding, output_padding=[0, 1, 1], stride=[1, 2, 2])
        self.Conv3d2UpIn = nn.Conv3d(filters * 2, filters, self.conv_ks, padding=padding)
        self.Conv3d2UpList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        # 3nd
        filters = 2 ** 2 * self.conv_filter_root
        self.Conv3d3DownIn = nn.Conv3d(2 ** 1 * self.conv_filter_root, filters, self.conv_ks, padding=padding)
        self.Conv3d3DownList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        self.Conv3dTranspose3 = nn.ConvTranspose3d(2 ** 3 * self.conv_filter_root, filters, self.conv_ks, padding=padding, output_padding=[0, 1, 1], stride=[1, 2, 2])
        self.Conv3d3UpIn = nn.Conv3d(filters * 2, filters, self.conv_ks, padding=padding)
        self.Conv3d3UpList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        # 4nd
        filters = 2 ** 3 * self.conv_filter_root
        self.Conv3d3DownIn = nn.Conv3d(2 ** 2 * self.conv_filter_root, filters, self.conv_ks, padding=padding)
        self.Conv3d3DownList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        self.Conv3dTranspose4 = nn.ConvTranspose3d(2 ** 4 * self.conv_filter_root, filters, self.conv_ks, padding=padding, output_padding=[0, 1, 1], stride=[1, 2, 2])
        self.Conv3d3UpIn = nn.Conv3d(filters * 2, filters, self.conv_ks, padding=padding)
        self.Conv3d3UpList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=padding) for i in range(self.conv_times)])

        # The Bottom
        filters = 2 ** 4 * self.conv_filter_root
        self.Conv3d5DownIn = nn.Conv3d(2 ** 3 * self.conv_filter_root, filters, self.conv_ks, padding=int((self.conv_ks - 1) / 2))
        self.Conv3d5DownList = nn.ModuleList([nn.Conv3d(filters, filters, self.conv_ks, padding=int((self.conv_ks - 1) / 2)) for i in range(self.conv_times)])

        ##### Not Trainable #####
        self.Dropout = nn.Dropout3d(0.1)
        self.Maxpool = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.Relu = nn.ReLU()

    def forward(self, x):
        layers_storage = []

        # 1st
        x = self.Dropout(self.Relu(self.Conv3d1DownIn(x)))
        for i in range(self.conv_times):
            x = self.Dropout(self.Relu(self.Conv3d1DownList[i](x)))

        layers_storage.append(x)
        x = self.Maxpool(x)

        # 2nd
        x = self.Dropout(self.Relu(self.Conv3d2DownIn(x)))
        for i in range(self.conv_times):
            x = self.Dropout(self.Relu(self.Conv3d2DownList[i](x)))

        layers_storage.append(x)
        x = self.Maxpool(x)

        # 3rd
        x = self.Dropout(self.Relu(self.Conv3d3DownIn(x)))
        for i in range(self.conv_times):
            x = self.Dropout(self.Relu(self.Conv3d3DownList[i](x)))

        layers_storage.append(x)
        x = self.Maxpool(x)

        # 4nd
        x = self.Dropout(self.Relu(self.Conv3d4DownIn(x)))
        for i in range(self.conv_times):
            x = self.Dropout(self.Relu(self.Conv3d4DownList[i](x)))

        layers_storage.append(x)
        x = self.Maxpool(x)

        # Bottom

        return x


class Net2D(TFBase):
    def __init__(self, config):
        self.config = config
        input_channel = int(config['2d-unet']['input_channel'])
        output_channel = int(config['2d-unet']['output_channel'])
        super().__init__(input_shape=[None, None, None, input_channel], output_shape=[None, None, None, output_channel])

    def get_train_op(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def get_loss(self):
        return tf.losses.mean_squared_error(self.y_output, self.y_gt)

    def get_net_output(self):

        def conv2d_bn_relu_dropout(input_, conv2d_kernel_size, conv2d_filters):
            result_ = tf.layers.conv2d(inputs=input_, filters=conv2d_filters, kernel_size=conv2d_kernel_size,
                                       padding='same')
            result_ = tf.nn.relu(result_)

            result_ = tf.layers.dropout(result_, rate=self.dropout_rate, training=self.dropout_training)
            return result_

        def conv2d_transpose_bn_relu_dropout(input_, conv2d_filters):
            result_ = tf.layers.conv2d_transpose(inputs=input_, filters=conv2d_filters, kernel_size=(2, 2),
                                                 strides=(2, 2), padding='same')
            result_ = tf.nn.relu(result_)

            result_ = tf.layers.dropout(result_, rate=self.dropout_rate, training=self.dropout_training)
            return result_

        conv_kernel_size = int(self.config['2d-unet']['conv_kernel_size'])
        conv_filters_root = int(self.config['2d-unet']['conv_filters_root'])
        conv_times = int(self.config['2d-unet']['conv_times'])
        up_or_down_times = int(self.config['2d-unet']['up_or_down_times'])

        layers_storage = []

        # Build Network
        result = conv2d_bn_relu_dropout(self.x, conv_kernel_size, conv_filters_root)
        print("Filters: %d" % conv_filters_root)

        for layer in range(up_or_down_times):

            conv_filters = 2 ** layer * conv_filters_root
            print("Filters: %d" % conv_filters)

            for conv_iter in range(0, conv_times):
                result = conv2d_bn_relu_dropout(result, conv_kernel_size, conv_filters)

            layers_storage.append(result)
            result = tf.layers.max_pooling2d(result, (2, 2), (2, 2))

        for layer in range(up_or_down_times - 1, -1, -1):

            conv_filters = 2 ** layer * conv_filters_root
            print("Filters: %d" % conv_filters)

            result = conv2d_transpose_bn_relu_dropout(result, conv_filters)
            result = tf.concat([result, layers_storage[layer]], -1)

            for conv_iter in range(0, conv_times):
                result = conv2d_bn_relu_dropout(result, conv_kernel_size, conv_filters)

        result = tf.layers.conv2d(result, 1, 1, padding='same')
        print("Filters: %d" % 1)

        return result


class Net3D(TFBase):
    def __init__(self, config, input_shape=None, output_shape=None):
        self.config = config

        if (input_shape is None) and (output_shape is None):
            input_shape = np.fromstring(self.config["3d-unet"]["input_shape"], dtype=np.int, sep=',')
            output_shape = np.fromstring(self.config["3d-unet"]["output_shape"], dtype=np.int, sep=',')

        super().__init__(input_shape=[None, input_shape[0], input_shape[1], input_shape[2], input_shape[3]],
                         output_shape=[None, output_shape[0], output_shape[1], output_shape[2], output_shape[3]])

        #  Shape = [Batch Depth X Y Channel]

    def get_train_op(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def get_loss(self):
        return tf.losses.mean_squared_error(self.y_output, self.y_gt)

    def get_net_output(self):

        def conv3d_bn_relu_dropout(input_, conv3d_kernel_size, conv3d_filters):
            result_ = tf.layers.conv3d(inputs=input_, filters=conv3d_filters, kernel_size=conv3d_kernel_size,
                                       padding='same')
            result_ = tf.nn.relu(result_)

            result_ = tf.layers.dropout(result_, rate=self.dropout_rate, training=self.dropout_training)
            return result_

        def conv3d_transpose_bn_relu_dropout(input_, conv3d_filters):
            result_ = tf.layers.conv3d_transpose(inputs=input_, filters=conv3d_filters, kernel_size=(2, 2, 2),
                                                 strides=(1, 2, 2), padding='same')
            result_ = tf.nn.relu(result_)

            result_ = tf.layers.dropout(result_, rate=self.dropout_rate, training=self.dropout_training)
            return result_

        conv_kernel_size = int(self.config['3d-unet']['conv_kernel_size'])
        conv_filters_root = int(self.config['3d-unet']['conv_filters_root'])
        conv_times = int(self.config['3d-unet']['conv_times'])
        up_or_down_times = int(self.config['3d-unet']['up_or_down_times'])

        layers_storage = []

        # Build Network
        result = conv3d_bn_relu_dropout(self.x, conv_kernel_size, conv_filters_root)
        print("Filters: %d" % conv_filters_root)
        print("Layer: ", result)

        for layer in range(up_or_down_times):

            conv_filters = 2 ** layer * conv_filters_root
            print("Filters: %d" % conv_filters)

            for conv_iter in range(0, conv_times):
                result = conv3d_bn_relu_dropout(result, conv_kernel_size, conv_filters)

            layers_storage.append(result)
            result = tf.layers.max_pooling3d(result, (1, 2, 2), (1, 2, 2))
            print("Layer: ", result)

        for layer in range(up_or_down_times - 1, -1, -1):

            conv_filters = 2 ** layer * conv_filters_root
            print("Filters: %d" % conv_filters)

            result = conv3d_transpose_bn_relu_dropout(result, conv_filters)
            result = tf.concat([result, layers_storage[layer]], -1)

            for conv_iter in range(0, conv_times):
                result = conv3d_bn_relu_dropout(result, conv_kernel_size, conv_filters)
            print("Layer: ", result)

        result = tf.layers.conv3d(result, 1, 1, padding='same')
        print("Filters: %d" % 1)
        print("Layer: ", result)

        return result


class ResNet3D(TFBase):

    def get_loss(self):
        y_gt_residual = self.y_gt - self.x
        network_output = self.y_output
        loss = tf.losses.mean_squared_error(network_output, y_gt_residual)

        self.y_output = network_output + self.x

        return loss

    def __init__(self, config):
        self.config = config

        input_shape = np.fromstring(self.config["3d-resunet"]["input_shape"], dtype=np.int, sep=',')
        output_shape = np.fromstring(self.config["3d-resunet"]["output_shape"], dtype=np.int, sep=',')

        super().__init__(input_shape=[None, input_shape[0], input_shape[1], input_shape[2], input_shape[3]],
                         output_shape=[None, output_shape[0], output_shape[1], output_shape[2], output_shape[3]])

        #  Shape = [Batch Depth X Y Channel]

    def get_net_output(self):

        def conv3d_bn_relu_dropout(input_, conv3d_kernel_size, conv3d_filters):
            result_ = tf.layers.conv3d(inputs=input_, filters=conv3d_filters, kernel_size=conv3d_kernel_size,
                                       padding='same')
            result_ = tf.nn.relu(result_)

            result_ = tf.layers.dropout(result_, rate=self.dropout_rate, training=self.dropout_training)
            return result_

        def conv3d_transpose_bn_relu_dropout(input_, conv3d_filters):
            result_ = tf.layers.conv3d_transpose(inputs=input_, filters=conv3d_filters, kernel_size=(2, 2, 2),
                                                 strides=(1, 2, 2), padding='same')
            result_ = tf.nn.relu(result_)

            result_ = tf.layers.dropout(result_, rate=self.dropout_rate, training=self.dropout_training)
            return result_

        conv_kernel_size = int(self.config['3d-resunet']['conv_kernel_size'])
        conv_filters_root = int(self.config['3d-resunet']['conv_filters_root'])
        conv_times = int(self.config['3d-resunet']['conv_times'])
        up_or_down_times = int(self.config['3d-resunet']['up_or_down_times'])

        layers_storage = []

        # Build Network
        result = conv3d_bn_relu_dropout(self.x, conv_kernel_size, conv_filters_root)
        print("Filters: %d" % conv_filters_root)
        print("Layer: ", result)

        for layer in range(up_or_down_times):

            conv_filters = 2 ** layer * conv_filters_root
            print("Filters: %d" % conv_filters)

            for conv_iter in range(0, conv_times):
                result = conv3d_bn_relu_dropout(result, conv_kernel_size, conv_filters)

            layers_storage.append(result)
            result = tf.layers.max_pooling3d(result, (1, 2, 2), (1, 2, 2))
            print("Layer: ", result)

        for layer in range(up_or_down_times - 1, -1, -1):

            conv_filters = 2 ** layer * conv_filters_root
            print("Filters: %d" % conv_filters)

            result = conv3d_transpose_bn_relu_dropout(result, conv_filters)
            result = tf.concat([result, layers_storage[layer]], -1)

            for conv_iter in range(0, conv_times):
                result = conv3d_bn_relu_dropout(result, conv_kernel_size, conv_filters)
            print("Layer: ", result)

        result = tf.layers.conv3d(result, 1, 1, padding='same')
        print("Filters: %d" % 1)
        print("Layer: ", result)

        return result
