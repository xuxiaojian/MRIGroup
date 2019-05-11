import configparser
import numpy as np
from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, Input, MaxPool3D, Concatenate, BatchNormalization, LeakyReLU
from tensorflow.python.keras.models import Model
from datasets.base import DatasetBase
import tensorflow as tf
from .base import TFNetBase, TensorboardXCallback
import logging
from . import utilities


class UNet3dGAN(object):
    def __init__(self, config: configparser.ConfigParser, config_section: str = 'UNet3dGAN'):
        self.config = config
        self.config_section = config_section

        self.input_shape = tuple(np.fromstring(self.config[self.config_section]['input_shape'], dtype=np.int, sep=',').tolist())
        self.output_shape = tuple(np.fromstring(self.config[self.config_section]['output_shape'], dtype=np.int, sep=',').tolist())

        self.generator = self.g_net()
        self.discriminator = self.d_net()

        self.generator.summary()
        self.discriminator.summary()

        self.metrics_name = ['g_loss', 'd_loss', 'psnr', 'ssim']

    @staticmethod
    def compute_metrics(predict, y):
        return tf.reduce_mean(tf.image.psnr(predict, y, max_val=1)).numpy(), tf.reduce_mean(tf.image.ssim(predict, y, max_val=1)).numpy()

    def train(self, train_dataset: DatasetBase, valid_dataset: DatasetBase = None):
        batch_size = int(self.config['Train']['batch_size'])
        train_epoch = int(self.config['Train']['train_epoch'])
        save_epoch = int(self.config['Train']['save_epoch'])

        root_path = self.config['Setting']['experiment_folder'] + self.config['Setting']['train_folder'] + '/'
        model_path, validation_path = TFNetBase.new_train_folders(code_path=self.config['Setting']['code_folder'], root_path=root_path)
        callback = TensorboardXCallback(config=self.config, output_path=root_path, validation_path=validation_path,
                                        train_dataset=train_dataset, valid_dataset=valid_dataset)

        g_learning_rate = float(self.config[self.config_section]['g_lr'])
        d_learning_rate = float(self.config[self.config_section]['d_lr'])

        optimizer_gen = tf.train.AdamOptimizer(g_learning_rate)
        optimizer_dis = tf.train.AdamOptimizer(d_learning_rate)

        loss_fn_mse = tf.keras.losses.MeanSquaredError()

        real_patch = tf.ones((batch_size,) + self.discriminator.output_shape[1:])
        fake_patch = tf.zeros((batch_size,) + self.discriminator.output_shape[1:])

        callback.on_train_begin()
        save_compared_metrics = np.zeros(shape=(len(self.metrics_name),))
        for now_epoch in range(train_epoch):

            train_metrics = np.zeros(shape=(len(self.metrics_name),))
            train_iter = 0
            for now_iter, (x, y) in enumerate(train_dataset.tf_dataset.batch(batch_size)):
                train_iter += 1

                with tf.GradientTape() as disc_tape:
                    logits_real = loss_fn_mse(self.discriminator(y, training=True), real_patch)
                    logits_fake = loss_fn_mse(self.discriminator(self.generator(x, training=True), training=True), fake_patch)
                    loss_dis = 0.5 * (logits_real + logits_fake)

                grads_dis = disc_tape.gradient(loss_dis, self.discriminator.variables)
                optimizer_dis.apply_gradients(zip(grads_dis, self.discriminator.variables))

                with tf.GradientTape() as gen_tape:
                    loss_gen = loss_fn_mse(self.discriminator(self.generator(x, training=True), training=True), real_patch)
                    loss_gen = (loss_gen + 4 * (loss_fn_mse(self.generator(x, training=True), y))) / 5

                grads_gen = gen_tape.gradient(loss_gen, self.generator.variables)
                optimizer_gen.apply_gradients(zip(grads_gen, self.generator.variables))

                predict = self.generator(x, training=False)
                metrics = (loss_gen.numpy(), loss_dis.numpy()) + self.compute_metrics(predict, y)
                logging.root.info('[Epoch %d Iter %d] Train Batch-Output: ' % (now_epoch, now_iter) + str(metrics))

                callback.on_train_batch_end(metrics, self.metrics_name)
                train_metrics += metrics

            train_metrics /= train_iter
            logging.root.info('[Epoch %d] Train Epoch-Output: ' % now_epoch + str(train_metrics))

            valid_metrics = np.zeros(shape=(len(self.metrics_name),))
            valid_iter = 0
            for now_iter, (x, y) in enumerate(valid_dataset.tf_dataset.batch(batch_size)):
                valid_iter += 1

                logits_real = loss_fn_mse(self.discriminator(y, training=False), real_patch)
                logits_fake = loss_fn_mse(self.discriminator(self.generator(x, training=False), training=False), fake_patch)
                loss_dis = 0.5 * (logits_real + logits_fake)

                loss_gen = loss_fn_mse(self.discriminator(self.generator(x, training=False), training=False), real_patch)
                loss_gen = (loss_gen + 4 * (loss_fn_mse(self.generator(x, training=False), y))) / 5

                predict = self.generator(x, training=False)
                metrics = (loss_gen.numpy(), loss_dis.numpy()) + self.compute_metrics(predict, y)
                valid_metrics += metrics

            valid_metrics /= valid_iter
            logging.root.info('[Epoch %d] Valid Epoch-Output: ' % now_epoch + str(valid_metrics))

            x, y = train_dataset.tf_sample.make_one_shot_iterator().get_next()
            train_pre = self.generator(x, training=False).numpy()

            x, y = valid_dataset.tf_sample.make_one_shot_iterator().get_next()
            valid_pre = self.generator(x, training=False).numpy()

            callback.on_epoch_end(train_pre, valid_pre, train_metrics, valid_metrics, self.metrics_name)

            if (now_epoch + 1) % save_epoch == 0:
                self.save(path=model_path + 'epoch_%d/' % (now_epoch + 1))
            for i in range(len(self.metrics_name)):
                if valid_metrics[i] > save_compared_metrics[i]:
                    logging.root.info('Found Better in ' + self.metrics_name[i] + '. From' + str(save_compared_metrics[i]) + ' to ' + str(valid_metrics[i]))
                    save_compared_metrics[i] = valid_metrics[i]

                    self.save(path=model_path + 'best_' + self.metrics_name[i] + '/')

    def d_net(self):

        kernel_size = int(self.config[self.config_section]['d_kernel_size'])
        filters_root = int(self.config[self.config_section]['d_filters_root'])
        layers = int(self.config[self.config_section]['d_layers'])

        def d_block(input_, filter_, stride):
            output_ = Conv3D(filters=filter_, kernel_size=kernel_size, padding='same', strides=(1, stride, stride))(input_)
            output_ = BatchNormalization()(output_)
            output_ = LeakyReLU(0.2)(output_)
            return output_

        net_input = Input(shape=self.input_shape)
        net = d_block(net_input, filters_root, stride=2)

        for i in range(layers // 2):
            filters = 2 ** (i + 1) * filters_root
            net = d_block(net, filters, stride=2)

        for i in range(layers // 2 - 1, -1, -1):
            filters = 2 ** (i + 1) * filters_root
            net = d_block(net, filters, stride=1)

        net = Conv3D(filters=1, kernel_size=kernel_size, padding='same')(net)

        return Model(inputs=net_input, outputs=net)

    def g_net(self):
        kernel_size = int(self.config[self.config_section]['g_kernel_size'])
        filters_root = int(self.config[self.config_section]['g_filters_root'])
        conv_times = int(self.config[self.config_section]['g_conv_times'])
        up_down_times = int(self.config[self.config_section]['g_up_down_times'])

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
        net_input = Input(shape=self.input_shape)
        net = conv3d_relu_dropout(net_input, filters_root, kernel_size)

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

        net = Conv3D(filters=1, kernel_size=1, padding='same', activation='tanh')(net)

        return Model(inputs=net_input, outputs=net)

    def save(self, path):
        utilities.new_folder(path)
        self.generator.save(path + 'model.h5')

    def load(self, path):
        pass
