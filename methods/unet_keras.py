from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, Input, MaxPool3D, Concatenate
from tensorflow.python.keras.models import Model
import numpy as np
from datasets.base import DatasetBase
import tensorflow as tf
from .base import TFNetBase, TensorboardXCallback
import configparser


class KerasCallBack(tf.keras.callbacks.Callback):
    def __init__(self, config: configparser.ConfigParser,
                 output_path, validation_path,
                 train_dataset: DatasetBase, valid_dataset: DatasetBase):
        super().__init__()
        self.tb_writer = TensorboardXCallback(config, output_path, validation_path, train_dataset, valid_dataset)
        self.metrics_name = ['loss', 'psnr', 'ssim']

    def on_train_begin(self, logs=None):
        self.tb_writer.on_train_begin()

    def on_train_batch_end(self, batch, logs=None):
        metrics = [logs['loss'], logs['psnr_tf'], logs['ssim_tf']]
        self.tb_writer.on_train_batch_end(metrics, self.metrics_name)

    def on_epoch_end(self, epoch, logs=None):
        train_pre = self.model.predict(self.tb_writer.train_x, verbose=1)
        valid_pre = self.model.predict(self.tb_writer.valid_x, verbose=1)

        train_metrics = [logs['loss'], logs['psnr_tf'], logs['ssim_tf']]
        valid_metrics = [logs['val_loss'], logs['val_psnr_tf'], logs['val_ssim_tf']]

        self.tb_writer.on_epoch_end(train_pre, valid_pre, train_metrics, valid_metrics, self.metrics_name)


class UNetKeras(object):
    def __init__(self, config, config_section: str = 'UNet3D'):
        self.config = config
        self.config_section = config_section

        self.input_shape = tuple(np.fromstring(self.config[self.config_section]['input_shape'], dtype=np.int, sep=',').tolist())
        self.output_shape = tuple(np.fromstring(self.config[self.config_section]['output_shape'], dtype=np.int, sep=',').tolist())

        self.model = self.get_model()
        self.model.summary()

    def train(self, train_dataset: DatasetBase, valid_dataset: DatasetBase = None):
        batch_size = int(self.config['Train']['batch_size'])
        learning_rate = float(self.config['Train']['learning_rate'])
        train_epoch = int(self.config['Train']['train_epoch'])
        save_epoch = int(self.config['Train']['save_epoch'])

        root_path = self.config['Setting']['experiment_folder'] + self.config['Setting']['train_folder'] + '/'
        model_path, validation_path = TFNetBase.new_train_folders(code_path=self.config['Setting']['code_folder'],root_path=root_path)
        tf.keras.utils.plot_model(self.model, to_file=root_path + 'model.png', show_shapes=True)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss=tf.keras.losses.mean_squared_error,
                           metrics=[self.psnr_tf, self.ssim_tf])

        self.model.fit(train_dataset.tf_dataset.batch(batch_size),
                       epochs=train_epoch,
                       steps_per_epoch=train_dataset.dataset_len() // batch_size,
                       validation_data=valid_dataset.tf_dataset.batch(batch_size),
                       validation_steps=valid_dataset.dataset_len() // batch_size,
                       callbacks=[
                            KerasCallBack(config=self.config,
                                          output_path=root_path,
                                          validation_path=validation_path,
                                          train_dataset=train_dataset,
                                          valid_dataset=valid_dataset,),
                            tf.keras.callbacks.ModelCheckpoint(model_path + '{epoch:02d}.h5', verbose=1, period=save_epoch),
                            tf.keras.callbacks.ModelCheckpoint(model_path + 'best_psnr.h5', verbose=1, save_best_only=True, monitor='val_psnr_tf', mode='max'),
                            tf.keras.callbacks.ModelCheckpoint(model_path + 'best_ssim.h5', verbose=1, save_best_only=True, monitor='val_ssim_tf', mode='max')
                       ])

    def test(self, test_dataset: DatasetBase):
        pass

    def get_model(self):
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

        net = Conv3D(filters=1, kernel_size=1, padding='same')(net)

        model = Model(inputs=net_input, outputs=net)

        return model

    @staticmethod
    def psnr_tf(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)

    @staticmethod
    def ssim_tf(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1)
