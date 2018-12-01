from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
from collections import OrderedDict
import logging


class UNet(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    :param kwargs: args passed to create_net function.
    """

    def __init__(self, img_channels=3, truth_channels=3, cost="mean_squared_error", **kwargs):
        tf.reset_default_graph()

        # basic variables
        self.summaries = kwargs.get("summaries", True)
        self.img_channels = img_channels
        self.truth_channels = truth_channels

        # placeholders for input x and y
        self.x = tf.placeholder("float", shape=[None, None, None, img_channels])
        self.y = tf.placeholder("float", shape=[None, None, None, truth_channels])
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # reused variables
        self.nx = tf.shape(self.x)[1]
        self.ny = tf.shape(self.x)[2]
        self.num_examples = tf.shape(self.x)[0]

        # variables need to be calculated
        self.recons = self.layers(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
        self.loss = self._get_cost(cost)
        self.valid_loss = self._get_cost(cost)
        self.avg_psnr = self._get_measure('avg_psnr')
        self.valid_avg_psnr = self._get_measure('avg_psnr')

    def _get_measure(self, measure):
        total_pixels = self.nx * self.ny * self.truth_channels
        dtype = self.x.dtype
        flat_recons = tf.reshape(self.recons, [-1, total_pixels])
        flat_truths = tf.reshape(self.y, [-1, total_pixels])

        if measure == 'psnr':
            # mse are of the same length of the truths
            mse = mse_array(flat_recons, flat_truths, total_pixels)
            term1 = log(tf.constant(1, dtype), 10.)
            term2 = log(mse, 10.)
            psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
            result = psnr

        elif measure == 'avg_psnr':
            # mse are of the same length of the truths
            mse = mse_array(flat_recons, flat_truths, total_pixels)
            term1 = log(tf.constant(1, dtype), 10.)
            term2 = log(mse, 10.)
            psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
            avg_psnr = tf.reduce_mean(psnr)
            result = avg_psnr

        else:
            raise ValueError("Unknown measure: " % measure)

        return result

    def _get_cost(self, cost_name):

        total_pixels = self.nx * self.ny * self.truth_channels
        flat_recons = tf.reshape(self.recons, [-1, total_pixels])
        flat_truths = tf.reshape(self.y, [-1, total_pixels])
        if cost_name == "mean_squared_error":
            loss = tf.losses.mean_squared_error(flat_recons, flat_truths)
        else:
            raise ValueError("Unknown cost function: " % cost_name)

        return loss

    def predict(self, model_path, x_test, keep_prob, phase=False):
        """
        Uses the model to create a prediction for the given data

        :param phase: training of Batch Normalization layers
        :param keep_prob:
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            prediction = sess.run(self.recons, feed_dict={self.x: x_test,
                                                          self.keep_prob: keep_prob,
                                                          self.phase: phase})  # set phase to False for every prediction
            # define operation
        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

    def layers(self, x, keep_prob, phase, img_channels, truth_channels, layers=3, conv_times=3, features_root=16,
               filter_size=3, pool_size=2, summaries=True):
        """
        Creates a new convolutional unet for the given parametrization.

        :param conv_times:
        :param truth_channels:
        :param phase:
        :param x: input tensor, shape [?,nx,ny,img_channels]
        :param keep_prob: dropout probability tensor
        :param img_channels: number of channels in the input image
        :param layers: number of layers in the net
        :param features_root: number of features in the first layer
        :param filter_size: size of the convolution filter
        :param pool_size: size of the max pooling operation
        :param summaries: Flag if summaries should be created
        """

        logging.info(
            "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, "
            "pool size: {pool_size}x{pool_size}".format(
                layers=layers,
                features=features_root,
                filter_size=filter_size,
                pool_size=pool_size))

        # Placeholder for the input image
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, img_channels]))

        pools = OrderedDict()  # pooling layers
        deconvs = OrderedDict()  # deconvolution layer
        dw_h_convs = OrderedDict()  # down-side convs
        up_h_convs = OrderedDict()  # up-side convs

        # conv the input image to desired feature maps
        in_node = conv2d_bn_relu(x_image, filter_size, features_root, keep_prob, phase, 'conv2feature_roots')

        # Down layers
        for layer in range(0, layers):
            features = 2 ** layer * features_root
            with tf.variable_scope('down_layer_' + str(layer)):
                for conv_iter in range(0, conv_times):
                    scope = 'conv_bn_relu_{}'.format(conv_iter)
                    conv = conv2d_bn_relu(in_node, filter_size, features, keep_prob, phase, scope)
                    in_node = conv

                # store the intermediate result per layer
                dw_h_convs[layer] = in_node

                # down sampling
                if layer < layers - 1:
                    with tf.variable_scope('pooling'):
                        pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                        in_node = pools[layer]

        in_node = dw_h_convs[layers - 1]

        # Up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            with tf.variable_scope('up_layer_' + str(layer)):
                with tf.variable_scope('unsample_concat_layer'):
                    # number of features = lower layer's number of features
                    h_deconv = deconv2d_bn_relu(in_node, filter_size, features // 2, pool_size, keep_prob, phase,
                                                'unsample_layer')
                    h_deconv_concat = concat(dw_h_convs[layer], h_deconv)
                    deconvs[layer] = h_deconv_concat
                    in_node = h_deconv_concat

                for conv_iter in range(0, conv_times):
                    scope = 'conv_bn_relu_{}'.format(conv_iter)
                    conv = conv2d_bn_relu(in_node, filter_size, features // 2, keep_prob, phase, scope)
                    in_node = conv

                up_h_convs[layer] = in_node

        in_node = up_h_convs[0]

        # Output with residual
        with tf.variable_scope("conv2d_1by1"):
            output = conv2d(in_node, 1, truth_channels, 'conv2truth_channels')
            up_h_convs["out"] = output

        if summaries:

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k, get_image_summary(pools[k]))

            for k in deconvs.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(deconvs[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

        return output

# Convient Layer Structure Function


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def log(x, base):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def rescale(array_x):  # convert to [0,1]
    amax = tf.reduce_max(array_x, axis=1, keep_dims=True)
    amin = tf.reduce_min(array_x, axis=1, keep_dims=True)
    rescaled = array_x - amin
    rescaled = rescaled / amax
    return rescaled


# receives an array of images and return the mse per image.
# size ~ num of pixels in the img
def mse_array(array_x, array_y, size):
    rescale_x = array_x
    rescale_y = array_y
    se = tf.reduce_sum(tf.squared_difference(rescale_x, rescale_y), 1)
    inv_size = tf.to_float(1 / size)
    return tf.scalar_mul(inv_size, se)


def conv2d_bn_relu(x, w_size, num_outputs, keep_prob_, phase, scope):  # output size should be the same.
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                       activation_fn=tf.nn.relu,  # elu is an alternative
                                       normalizer_fn=tf.layers.batch_normalization,
                                       normalizer_params={'training': phase},
                                       scope=scope)

    return tf.nn.dropout(conv_2d, keep_prob_)


def deconv2d_bn_relu(x, w_size, num_outputs, stride, keep_prob_, phase, scope):
    conv_2d = tf.contrib.layers.conv2d_transpose(x, num_outputs, w_size,
                                                 stride=stride,
                                                 activation_fn=tf.nn.relu,  # elu is an alternative
                                                 normalizer_fn=tf.layers.batch_normalization,
                                                 normalizer_params={'training': phase},
                                                 scope=scope)

    return tf.nn.dropout(conv_2d, keep_prob_)


def conv2d_bn(x, w_size, num_outputs, phase, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                       activation_fn=None,
                                       normalizer_fn=tf.layers.batch_normalization,
                                       normalizer_params={'training': phase},
                                       scope=scope)
    return conv_2d


def conv2d(x, w_size, num_outputs, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       scope=scope)
    return conv_2d


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')


def concat(x1, x2):
    return tf.concat([x1, x2], 3)