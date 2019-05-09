from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, MaxPool3D, Concatenate, TimeDistributed
import tensorflow as tf
from methods.base import TFNetBase
import numpy as np
from datasets.base import DatasetBase


class UNet3DLSTM(TFNetBase):
    def __init__(self, config):
        super().__init__(config=config, config_section='UNet3DLSTM')

    def get_output(self):
        kernel_size = int(self.config[self.config_section]['kernel_size'])
        filters_root = int(self.config[self.config_section]['filters_root'])
        conv_times = int(self.config[self.config_section]['conv_times'])
        up_down_times = int(self.config[self.config_section]['up_down_times'])

        def conv3d_relu_dropout_parallel_with_lstm(input_, input_shape: list, conv_filter, conv_ks, scope_index):
            assert input_shape.__len__() == 4
            # noinspection SpellCheckingInspection
            with tf.variable_scope(name_or_scope='conv3dlstmcell_%d' % scope_index):
                rnn_cell_fw = tf.contrib.rnn.Conv3DLSTMCell(
                    input_shape=input_shape,
                    output_channels=conv_filter,
                    kernel_shape=[conv_ks, conv_ks, conv_ks],
                    skip_connection=False
                )

                rnn_cell_bw = tf.contrib.rnn.Conv3DLSTMCell(
                    input_shape=input_shape,
                    output_channels=conv_filter,
                    kernel_shape=[conv_ks, conv_ks, conv_ks],
                    skip_connection=False
                )

                # noinspection PyUnresolvedReferences
                output_, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, input_, dtype=tf.float32)

            output_ = tf.concat(output_, -1)
            output_ = TimeDistributed(ReLU())(output_)
            output_ = TimeDistributed(Dropout(rate=0.1))(output_)

            return output_

        def conv3d_relu_dropout_parallel_without_lstm(input_, conv_filter, conv_ks):
            output_ = TimeDistributed(Conv3D(filters=conv_filter, kernel_size=conv_ks, padding='same'))(input_)
            output_ = TimeDistributed(ReLU())(output_)
            output_ = TimeDistributed(Dropout(rate=0.1))(output_)

            return output_

        def conv3d_transpose_relu_dropout_parallel_without_lstm(input_, conv_filter, conv_ks):
            output_ = TimeDistributed(Conv3DTranspose(filters=conv_filter, kernel_size=conv_ks, padding='same', strides=(1, 2, 2)))(input_)
            output_ = TimeDistributed(ReLU())(output_)
            output_ = TimeDistributed(Dropout(rate=0.1))(output_)
            return output_

        _, depth, width, height, channel = self.input_shape
        width_min = int(width / (2 ** up_down_times))
        height_min = int(height / (2 ** up_down_times))
        skip_layers_storage = []

        net = conv3d_relu_dropout_parallel_without_lstm(self.x, filters_root, kernel_size)

        for layer in range(up_down_times):
            filters = 2 ** layer * filters_root
            for i in range(0, conv_times):
                net = conv3d_relu_dropout_parallel_without_lstm(net, filters, kernel_size)

            skip_layers_storage.append(net)
            net = TimeDistributed(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))(net)

        filters = 2 ** up_down_times * filters_root
        for i in range(0, conv_times):
            net = conv3d_relu_dropout_parallel_with_lstm(net, [depth, width_min, height_min, channel], filters, kernel_size, scope_index=i)

        for layer in range(up_down_times - 1, -1, -1):

            filters = 2 ** layer * filters_root
            net = conv3d_transpose_relu_dropout_parallel_without_lstm(net, filters, kernel_size)
            net = Concatenate(axis=-1)([net, skip_layers_storage[layer]])
            for i in range(0, conv_times):
                net = conv3d_relu_dropout_parallel_without_lstm(net, filters, kernel_size)

        net = TimeDistributed(Conv3D(filters=1, kernel_size=1, padding='same'))(net)

        return net

    def save_test(self, metrics, outputs, dataset: DatasetBase, save_path):
        output_file = self.config['Test']['output_file']
        is_slim = bool(int(self.config['Test']['is_slim']))

        def transform(input_):
            output_ = np.squeeze(input_[:, 1])

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
