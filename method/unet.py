from method.tfbase import TFBase
import tensorflow as tf


class Net2D(TFBase):
    def __init__(self, config):
        self.config = config

        super().__init__(input_shape=[None, None, None, 1], output_shape=[None, None, None, 1])

    def get_train_op(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def get_loss(self):
        return tf.losses.mean_squared_error(self.y_output, self.y_gt)

    def get_net_output(self):

        def conv2d_bn_relu_dropout(input_, conv2d_kernel_size, conv2d_filters):
            result_ = tf.layers.conv2d(inputs=input_, filters=conv2d_filters, kernel_size=conv2d_kernel_size,
                                       padding='same')
            # result_ = tf.layers.batch_normalization(inputs=result_, training=self.bn_training)
            result_ = tf.nn.relu(result_)

            result_ = tf.layers.dropout(result_, rate=self.dropout_rate, training=self.dropout_training)
            return result_

        def conv2d_transpose_bn_relu_dropout(input_, conv2d_filters):
            result_ = tf.layers.conv2d_transpose(inputs=input_, filters=conv2d_filters, kernel_size=(2, 2),
                                                 strides=(2, 2), padding='same')
            # result_ = tf.layers.batch_normalization(inputs=result_, training=self.bn_training)
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
