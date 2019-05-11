from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, Input, MaxPool3D, Concatenate
from methods.base import TFNetBase
from datasets.base import DatasetBase
import numpy as np
import tensorflow as tf


class UNet3D(TFNetBase):
    def __init__(self, config):
        super().__init__(config=config, config_section='UNet3D')

    def get_output(self):
        kernel_size = int(self.config[self.config_section]['kernel_size'])
        filters_root = int(self.config[self.config_section]['filters_root'])
        conv_times = int(self.config[self.config_section]['conv_times'])
        up_down_times = int(self.config[self.config_section]['up_down_times'])

        def conv3d_relu_dropout(input_, filters_, kernel_size_):
            output_ = Conv3D(filters=filters_, kernel_size=kernel_size_, padding='same')(input_)
            output_ = ReLU()(output_)
            output_ = Dropout(rate=0.1)(output_)
            return output_

        def conv3d_transpose_relu_dropout(input_, filters_, kernel_size_):
            output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(1, 2, 2))(
                input_)
            output_ = ReLU()(output_)
            output_ = Dropout(rate=0.1)(output_)
            return output_

        skip_layers_storage = []

        # Build Network
        net = conv3d_relu_dropout(self.x, filters_root, kernel_size)

        for layer in range(up_down_times):
            filters = 2 ** layer * filters_root
            for i in range(0, conv_times):
                net = conv3d_relu_dropout(net, filters, kernel_size)

            skip_layers_storage.append(net)
            net = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(net)

        filters = 2 ** up_down_times * filters_root
        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size)

        for layer in range(up_down_times - 1, -1, -1):

            filters = 2 ** layer * filters_root
            net = conv3d_transpose_relu_dropout(net, filters, kernel_size)
            net = Concatenate(axis=-1)([net, skip_layers_storage[layer]])

            for i in range(0, conv_times):
                net = conv3d_relu_dropout(net, filters, kernel_size)

        net = Conv3D(filters=1, kernel_size=1, padding='same')(net)

        return net

    def save_test(self, metrics, outputs, dataset: DatasetBase, save_path):
        output_file = self.config['Test']['output_file']
        is_slim = bool(int(self.config['Test']['is_slim']))

        def transform(input_):
            output_ = input_
            if is_slim:
                output_ = np.squeeze(output_[:, 0])
            else:
                width = output_.shape[-1]
                height = output_.shape[-2]
                output_.shape = [-1, height, width]

            return output_

        predict = np.concatenate(outputs, 0)
        with tf.Session() as sess:
            x, y = sess.run(dataset.tf_dataset.batch(dataset.dataset_len()).make_one_shot_iterator().get_next())

        x = transform(x)
        y = transform(y)
        predict = transform(predict)

        write_path = save_path + output_file

        self.write_test(x, y, predict, write_path)
