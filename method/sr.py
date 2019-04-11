from method.tfbase import TFBase
import tensorflow as tf
import numpy as np


class SRNet3D(TFBase):
    def __init__(self, config):
        self.config = config
        input_shape = np.fromstring(self.config["3d-sr"]["input_shape"], dtype=np.int, sep=',')
        output_shape = np.fromstring(self.config["3d-sr"]["output_shape"], dtype=np.int, sep=',')

        super().__init__(input_shape=[None, input_shape[0], input_shape[1], input_shape[2], input_shape[3]],
                         output_shape=[None, output_shape[0], output_shape[1], output_shape[2], output_shape[3]])

    def get_net_output(self):

        def block(input_, filters_, kernel_size):
            result_ = tf.layers.conv3d(inputs=input_, filters=filters_, kernel_size=kernel_size, padding='same')
            result_ = tf.nn.relu(result_)
            result_ = tf.layers.conv3d(inputs=result_, filters=filters_, kernel_size=kernel_size, padding='same')
            return result_ + input_

        filters = int(self.config['3d-sr']['filters'])
        block_kernel_size = int(self.config['3d-sr']['block_kernel_size'])
        block_nums = int(self.config['3d-sr']['block_nums'])

        io_kernel_size = int(self.config['3d-sr']['io_kernel_size'])

        net = tf.layers.conv3d(inputs=self.x, filters=filters, kernel_size=io_kernel_size, padding='same')
        block_input = net

        for i in range(block_nums):
            net = block(net, filters, block_kernel_size)

        net = tf.layers.conv3d(inputs=net, filters=filters, kernel_size=block_kernel_size, padding='same')
        net = net + block_input
        net = tf.layers.conv3d(inputs=net, filters=filters, kernel_size=block_kernel_size, padding='same')
        net = tf.layers.conv3d(inputs=net, filters=1, kernel_size=io_kernel_size, padding='same')

        return net


class SRNet2D(TFBase):
    def __init__(self, config):
        self.config = config
        input_channel = int(config['2d-sr']['input_channel'])
        output_channel = int(config['2d-sr']['output_channel'])
        super().__init__(input_shape=[None, None, None, input_channel], output_shape=[None, None, None, output_channel])

    def get_net_output(self):

        def block(input_, filters_, kernel_size):
            result_ = tf.layers.conv2d(inputs=input_, filters=filters_, kernel_size=kernel_size, padding='same')
            result_ = tf.nn.relu(result_)
            result_ = tf.layers.conv2d(inputs=result_, filters=filters_, kernel_size=kernel_size, padding='same')
            return result_ + input_

        filters = int(self.config['2d-sr']['filters'])
        block_kernel_size = int(self.config['2d-sr']['block_kernel_size'])
        block_nums = int(self.config['2d-sr']['block_nums'])
        io_kernel_size = int(self.config['2d-sr']['io_kernel_size'])

        print(self.x)
        net = tf.layers.conv2d(inputs=self.x, filters=filters, kernel_size=io_kernel_size, padding='same')
        print(net)
        block_input = net

        for i in range(block_nums):
            net = block(net, filters, block_kernel_size)
            print(net)

        net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=block_kernel_size, padding='same')
        print(net)
        net = net + block_input
        net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=block_kernel_size, padding='same')
        print(net)
        net = tf.layers.conv2d(inputs=net, filters=1, kernel_size=io_kernel_size, padding='same')
        print(net)

        return net
