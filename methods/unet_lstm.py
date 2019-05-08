from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, MaxPool3D, Concatenate, TimeDistributed
import tensorflow as tf
import numpy as np
from datasets.base import DatasetBase
import tensorflow.keras.backend as k
import logging
from tensorboardX import SummaryWriter
import configparser
from methods import utilities
import scipy.io as sio


def new_train_folders(root_path, code_path):
    utilities.copytree_code(src_path=code_path, dst_path=root_path)
    utilities.set_logging(target_path=root_path)

    model_path = root_path + 'model/'
    validation_path = root_path + 'validation/'
    utilities.new_folder(model_path)
    utilities.new_folder(validation_path)

    return model_path, validation_path


class TensorboardXCallback(object):
    def __init__(self,
                 config: configparser.ConfigParser,
                 output_path, validation_path,
                 train_dataset: DatasetBase, valid_dataset: DatasetBase):

        super().__init__()

        self.writer = SummaryWriter(output_path)
        self.validation_path = validation_path

        self.global_batch = 0
        self.global_epoch = 0

        config_info = str()
        for section in config.sections():
            config_info = config_info + utilities.dict_to_markdown_table(config._sections[section], section)
        self.writer.add_text(tag='config', text_string=config_info, global_step=0)

        with tf.Session() as sess:
            self.train_x, self.train_y = sess.run(train_dataset.tf_sample.make_one_shot_iterator().get_next())
            self.valid_x, self.valid_y = sess.run(valid_dataset.tf_sample.make_one_shot_iterator().get_next())

    def on_train_begin(self):
        sio.savemat(self.validation_path + 'init.mat', {'x': self.valid_x, 'y': self.valid_y})

        for i in range(self.train_x.shape[0]):
            self.add_imgs(self.train_x[i], tag='train/%d_x' % i, step=0)
            self.add_imgs(self.train_y[i], tag='train/%d_y' % i, step=0)

            self.add_imgs(self.valid_x[i], tag='valid/%d_x' % i, step=0)
            self.add_imgs(self.valid_y[i], tag='valid/%d_y' % i, step=0)

    def on_train_batch_end(self, metrics: list, metrics_name: list):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='train_batch/' + metrics_name[i], scalar_value=metrics[i], global_step=self.global_batch)
        self.global_batch += 1

    def on_epoch_end(self, train_pre, valid_pre, train_metrics: list, valid_metrics: list, metrics_name: list):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='train_epoch/' + metrics_name[i], scalar_value=train_metrics[i], global_step=self.global_epoch)

        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='valid_epoch/' + metrics_name[i], scalar_value=valid_metrics[i], global_step=self.global_epoch)

        sio.savemat(self.validation_path + 'epoch.%d.mat' % self.global_epoch, {'predict': valid_pre})
        for i in range(train_pre.shape[0]):
            self.add_imgs(train_pre[i], tag='train/%d_predict' % i, step=self.global_epoch)
            self.add_imgs(valid_pre[i], tag='valid/%d_predict' % i, step=self.global_epoch)

        self.global_epoch += 1

    def add_imgs(self, imgs_input, tag, step):
        from skimage.color import gray2rgb

        channel = imgs_input.shape[-1]
        width = imgs_input.shape[-2]
        height = imgs_input.shape[-3]

        imgs_input.shape = [-1, height, width, channel]
        new_batch = imgs_input.shape[0]

        assert channel == 1 or channel == 3

        if channel == 1:
            imgs_output = np.zeros(shape=[new_batch, height, width, 3])
            for i in range(new_batch):
                imgs_output[i] = gray2rgb(np.squeeze(imgs_input[i]))
        else:
            imgs_output = imgs_input

        self.writer.add_images(tag=tag, img_tensor=imgs_output, global_step=step, dataformats='NHWC')


class UNet3DLSTM(object):
    def __init__(self, config):
        self.config = config
        self.config_section = 'UNet3DLSTM'

        self.input_shape = tuple(np.fromstring(self.config[self.config_section]['input_shape'], dtype=np.int, sep=',').tolist())
        self.output_shape = tuple(np.fromstring(self.config[self.config_section]['output_shape'], dtype=np.int, sep=',').tolist())

        self.dataset_iter = tf.data.Iterator.from_structure(output_types=(tf.float32, tf.float32),
                                                            output_shapes=((None, ) + self.input_shape, (None, ) + self.output_shape))
        self.x, self.y = self.dataset_iter.get_next()

        self.output = self.get_output()
        self.loss = self.get_loss()
        self.metrics, self.metrics_name = self.get_metrics()

    def train(self, train_dataset: DatasetBase, valid_dataset: DatasetBase = None):
        batch_size = int(self.config['Train']['batch_size'])
        learning_rate = float(self.config['Train']['learning_rate'])
        train_epoch = int(self.config['Train']['train_epoch'])
        save_epoch = int(self.config['Train']['save_epoch'])

        root_path = self.config['Setting']['experiment_folder'] + self.config['Setting']['train_folder'] + '/'
        model_path, validation_path = new_train_folders(code_path=self.config['Setting']['code_folder'], root_path=root_path)
        callback = TensorboardXCallback(config=self.config, output_path=root_path, validation_path=validation_path,
                                        train_dataset=train_dataset, valid_dataset=valid_dataset)

        train_op = self.get_train_op(learning_rate=learning_rate)
        save_compared_metrics = np.zeros(shape=(len(self.metrics_name), ))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            callback.on_train_begin()
            for now_epoch in range(train_epoch):

                sess.run(self.dataset_iter.make_initializer(train_dataset.tf_dataset.batch(batch_size=batch_size)))
                train_iter = train_dataset.dataset_len() // batch_size
                train_metrics = np.zeros(shape=(len(self.metrics_name), ))
                for now_iter in range(train_iter):
                    _, metrics = sess.run([train_op, self.metrics], feed_dict={k.learning_phase(): 1})
                    logging.root.info('[Epoch %d Iter %d (%d in Total)] Train Batch-Output: ' % (now_epoch, now_iter, train_iter) + str(metrics))
                    callback.on_train_batch_end(metrics, self.metrics_name)

                    train_metrics += metrics

                train_metrics /= train_iter
                logging.root.info('[Epoch %d] Train Epoch-Output: ' % now_epoch + str(train_metrics))

                sess.run(self.dataset_iter.make_initializer(valid_dataset.tf_dataset.batch(batch_size=batch_size)))
                valid_iter = valid_dataset.dataset_len() // batch_size
                valid_metrics = np.zeros(shape=(len(self.metrics_name),))
                for now_iter in range(valid_iter):
                    metrics = sess.run(self.metrics, feed_dict={k.learning_phase(): 0})

                    valid_metrics += metrics

                valid_metrics /= valid_iter
                logging.root.info('[Epoch %d] Valid Epoch-Output: ' % now_epoch + str(valid_metrics))

                sess.run(self.dataset_iter.make_initializer(train_dataset.tf_sample))
                train_pre = sess.run(self.output, feed_dict={k.learning_phase(): 0})

                sess.run(self.dataset_iter.make_initializer(valid_dataset.tf_sample))
                valid_pre = sess.run(self.output, feed_dict={k.learning_phase(): 0})

                callback.on_epoch_end(train_pre, valid_pre, train_metrics, valid_metrics, self.metrics_name)

                if (now_epoch + 1) % save_epoch == 0:
                    self.save(sess, path=model_path + 'epoch_%d/' % (now_epoch + 1))
                for i in range(len(self.metrics_name)):
                    if valid_metrics[i] > save_compared_metrics[i]:
                        logging.root.info('Found Better in ' + self.metrics_name[i] + '. From' + str(save_compared_metrics[i]) + ' to ' + str(valid_metrics[i]))
                        save_compared_metrics[i] = valid_metrics[i]

                        self.save(sess, path=model_path + 'best_' + self.metrics_name[i] + '/')

    @staticmethod
    def save(sess, path):
        utilities.new_folder(path)
        saver = tf.train.Saver()
        saver.save(sess, save_path=path + 'model.ckpt')

    @staticmethod
    def load(sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path + 'model.ckpt')

    def get_metrics(self):
        psnr = tf.reduce_mean(tf.image.psnr(self.output, self.y, 1))
        ssim = tf.reduce_mean(tf.image.ssim(self.output, self.y, 1))

        metrics = [self.loss, psnr, ssim]
        metrics_name = ["loss", "psnr", "ssim"]

        return metrics, metrics_name

    def get_loss(self):
        return tf.losses.mean_squared_error(self.y, self.output)

    def get_train_op(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(loss=self.loss)

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
