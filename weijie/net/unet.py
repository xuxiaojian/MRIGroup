from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
from collections import OrderedDict
import os
import shutil
import numpy as np
from util import imgProcess,dataLoader
from tensorboardX import SummaryWriter


class TFNetwork(object):

    def __init__(self, config):

        self.config = config
        self._set_parameter()

        self.x = tf.placeholder("float", shape=[None, None, None, self.img_channels])
        self.y = tf.placeholder("float", shape=[None, None, None, self.truth_channels])
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        self.recons = self._main_structure(self.x, self.phase, self.keep_prob)

        self.losses = tf.losses.mean_squared_error(self.recons, self.y)
        self.psnr = tf.image.psnr(self.recons, self.y, max_val=1)
        self.ssim = tf.image.ssim(self.recons, self.y, max_val=1)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses)

    def train(self, x_train, y_train, x_val, y_val, display=True):

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path, ignore_errors=True)
        os.makedirs(self.output_path)

        imgs_train = dataLoader.SimpleDataProvider(x_train, y_train)
        imgs_valid = dataLoader.SimpleDataProvider(x_val, y_val)
        x_valid, y_valid = imgs_valid(imgs_valid.file_count, fix=True)

        saver_model = tf.train.Saver()
        saver_tb = SummaryWriter(self.output_path)
        saver_tb.add_text(tag='config', text_string=self.config_info, global_step=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            def record_batch(loss, psnr, ssim, step):
                saver_tb.add_scalar(tag='train/loss', scalar_value=loss, global_step=step)
                saver_tb.add_scalar(tag='train/psnr', scalar_value=psnr.mean(), global_step=step)
                saver_tb.add_scalar(tag='train/ssim', scalar_value=ssim.mean(), global_step=step)

            def record_epoch(loss, psnr, ssim, vals, step, img_tag):
                saver_tb.add_scalar(tag='valid_avg/loss', scalar_value=loss, global_step=step)
                saver_tb.add_scalar(tag='valid_avg/psnr', scalar_value=psnr.mean(), global_step=step)
                saver_tb.add_scalar(tag='valid_avg/ssim', scalar_value=ssim.mean(), global_step=step)

                for i in range(vals.shape[0]):
                    saver_tb.add_scalar(tag='valid_%d/psnr' % i, scalar_value=psnr[i], global_step=step)
                    saver_tb.add_scalar(tag='valid_%d/ssim' % i, scalar_value=ssim[i], global_step=step)
                    val = np.reshape(vals[i, :, :, :], [vals.shape[1], vals.shape[2]])
                    val = imgProcess.normalize(val)
                    saver_tb.add_image(tag=('valid_%d/' % i) + img_tag,
                                       img_tensor=val, global_step=step)

                pass

            global_step = 0
            init_psnr, init_ssim, init_loss = sess.run([self.psnr, self.ssim, self.losses], feed_dict={
                self.x: x_valid, self.y: y_valid, self.keep_prob: 1, self.phase: False})
            record_epoch(init_loss, init_psnr, init_ssim, y_valid, global_step, 'ground-truth')
            record_epoch(init_loss, init_psnr, init_ssim, x_valid, global_step, 'prediction')

            batches = int(imgs_train.file_count / self.batch_size)
            for epoch in range(self.epochs):
                for batch in range(batches):

                    global_step += 1
                    batch_x, batch_y = imgs_train(self.batch_size)
                    sess.run([self.optimizer], feed_dict={
                            self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout, self.phase: True})

                    batch_loss, batch_psnr, batch_ssim = \
                        sess.run([self.losses, self.psnr, self.ssim], feed_dict={
                            self.x: batch_x, self.y: batch_y, self.keep_prob: 1, self.phase: False})

                    record_batch(batch_loss, batch_psnr, batch_ssim, global_step)

                    if display:
                        print('EPOCH: [%d], [%.3f] PERCENTAGE. LOSS: [%.4f], AVG_PSNR: [%.4f], AVG_SSIM: [%.4f]'
                              % (epoch+1, ((batch + 1) * 100) / batches, batch_loss, batch_psnr.mean(),
                                 batch_ssim.mean()))

                loss_val, psnr_val, ssim_val, valids = sess.run([self.losses, self.psnr, self.ssim, self.recons],
                                                                feed_dict={self.x: x_valid, self.y: y_valid,
                                                                           self.keep_prob: 1, self.phase: False
                                                                           })

                record_epoch(loss_val, psnr_val, ssim_val, valids, global_step, 'prediction')
                saver_model.save(sess, self.output_path + 'model.cpkt')

                if display:
                    print('EPOCH: [%d] DONE!!. IN VALIDATION, LOSS: [%.4f], AVG_PSNR: [%.4f], AVG_SSIM: [%.4f]'
                          % (epoch + 1, loss_val, psnr_val.mean(), ssim_val.mean()))

    def _main_structure(self, x, phase, keep_prob):

        def conv2d_bn_relu(x_, w_size, num_outputs, keep_prob_, phase_, scope_):  # output size should be the same.
            conv_2d = tf.contrib.layers.conv2d(x_, num_outputs, w_size,
                                               activation_fn=tf.nn.relu,  # elu is an alternative
                                               normalizer_fn=tf.layers.batch_normalization,
                                               normalizer_params={'training': phase_},
                                               scope=scope_)

            return tf.nn.dropout(conv_2d, keep_prob_)

        def deconv2d_bn_relu(x_, w_size, num_outputs, stride, keep_prob_, phase_, scope_):
            conv_2d = tf.contrib.layers.conv2d_transpose(x_, num_outputs, w_size,
                                                         stride=stride,
                                                         activation_fn=tf.nn.relu,  # elu is an alternative
                                                         normalizer_fn=tf.layers.batch_normalization,
                                                         normalizer_params={'training': phase_},
                                                         scope=scope_)

            return tf.nn.dropout(conv_2d, keep_prob_)

        def conv2d(x_, w_size, num_outputs, scope_):
            conv_2d = tf.contrib.layers.conv2d(x_, num_outputs, w_size,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               scope=scope_)
            return conv_2d

        def max_pool(x_, n):
            return tf.nn.max_pool(x_, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

        def concat(x1, x2):
            return tf.concat([x1, x2], 3)

        # Placeholder for the input image
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, self.img_channels]))

        pools = OrderedDict()  # pooling layers
        deconvs = OrderedDict()  # deconvolution layer
        dw_h_convs = OrderedDict()  # down-side convs
        up_h_convs = OrderedDict()  # up-side convs

        # conv the input image to desired feature maps
        in_node = conv2d_bn_relu(x_image, self.filter_size, self.features_root, keep_prob, phase, 'conv2feature_roots')

        # Down layers
        for layer in range(0, self.layers):
            features = 2 ** layer * self.features_root
            with tf.variable_scope('down_layer_' + str(layer)):
                for conv_iter in range(0, self.conv_times):
                    scope = 'conv_bn_relu_{}'.format(conv_iter)
                    conv = conv2d_bn_relu(in_node, self.filter_size, features, keep_prob, phase, scope)
                    in_node = conv

                # store the intermediate result per layer
                dw_h_convs[layer] = in_node

                # down sampling
                if layer < self.layers - 1:
                    with tf.variable_scope('pooling'):
                        pools[layer] = max_pool(dw_h_convs[layer], self.pool_size)
                        in_node = pools[layer]

        in_node = dw_h_convs[self.layers - 1]

        # Up layers
        for layer in range(self.layers - 2, -1, -1):
            features = 2 ** (layer + 1) * self.features_root
            with tf.variable_scope('up_layer_' + str(layer)):
                with tf.variable_scope('unsample_concat_layer'):
                    # number of features = lower layer's number of features
                    h_deconv = deconv2d_bn_relu(in_node, self.filter_size, features // 2,
                                                self.pool_size, keep_prob, phase,
                                                'unsample_layer')
                    h_deconv_concat = concat(dw_h_convs[layer], h_deconv)
                    deconvs[layer] = h_deconv_concat
                    in_node = h_deconv_concat

                for conv_iter in range(0, self.conv_times):
                    scope = 'conv_bn_relu_{}'.format(conv_iter)
                    conv = conv2d_bn_relu(in_node, self.filter_size, features // 2, keep_prob, phase, scope)
                    in_node = conv

                up_h_convs[layer] = in_node

        in_node = up_h_convs[0]

        # Output with residual
        with tf.variable_scope("conv2d_1by1"):
            output = conv2d(in_node, 1, self.truth_channels, 'conv2truth_channels')
            up_h_convs["out"] = output

        return output

    def _set_parameter(self):

        if self.config is None:
            print('ERROR!! PLEASE USE RIGHT CONFIGPARSER OBJECT')

        # Global
        self.output_path = self.config['GLOBAL']['output_path']

        # Training Paremeters
        self.batch_size = int(self.config['UNet']['batch_size'])
        self.epochs = int(self.config['UNet']['epochs'])
        self.learning_rate = float(self.config['UNet']['learning_rate'])
        self.dropout = float(self.config['UNet']['dropout'])

        # Network Paremeters
        self.img_channels = int(self.config['UNet']['img_channels'])
        self.truth_channels = int(self.config['UNet']['truth_channels'])
        self.layers = int(self.config['UNet']['layers'])
        self.conv_times = int(self.config['UNet']['conv_times'])
        self.features_root = int(self.config['UNet']['features_root'])
        self.filter_size = int(self.config['UNet']['filter_size'])
        self.pool_size = int(self.config['UNet']['pool_size'])

        config_info = open('./config.ini', 'r')
        self.config_info = config_info.read()
        config_info.close()