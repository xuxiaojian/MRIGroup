import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Add, PReLU, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import shutil
import tensorflow as tf
import scipy.io as sio
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os


class KesNetwork(object):

    def __init__(self, config, shape_train, shape_valid):

        self.set_parameter(config)

        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, max_val=1)

        self.net_train = self.build_network(shape_train)
        self.net_train.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr])
        self.net_train.summary()

        self.net_valid = self.build_network(shape_valid)
        self.net_valid.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                               metrics=[psnr])

    def train(self, data, valid):

        net_valid = self.net_valid
        validation_path = self.validation_path
        epoch_save_valid = self.epoch_save_valid

        class SaveValidation(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % epoch_save_valid == 0:
                    net_valid.set_weights(self.model.get_weights())
                    pre = net_valid.predict(valid[0])
                    sio.savemat(validation_path + '%d.mat' % (epoch + 1), {
                        'pre_sr_valid': pre,
                    })

            def on_train_begin(self, logs=None):
                # sio.savemat(validation_path + 'init.mat', {
                #     'x_noised_valid': valid[0],
                #     'x_sr_valid': valid[1],
                #     'y_valid': valid[2],
                # })
                sio.savemat(validation_path + 'init.mat', {
                    'x_sr_valid': valid[0],
                    'y_valid': valid[1],
                })

        # Train Network
        self.net_train.fit(x=data[0], y=data[1], batch_size=self.batch_size, epochs=self.epochs,
                           callbacks=[ModelCheckpoint(filepath=self.model_path + 'model_{epoch:02d}.h5',
                                                      period=self.epoch_save_model, save_weights_only=True),
                                      SaveValidation(),
                                      TensorBoard(log_dir=self.model_path)
                                      ])

    def build_network(self, shape_input, num_rblocks=16, num_filters=64):

        input_data = Input(shape=shape_input)
        net = Conv2D(filters=num_filters, kernel_size=9, padding='same', name='conv2d_input')(input_data)
        # kernel_size = 9 is learnt from SRResNet
        input_net = net

        for i in range(num_rblocks):
            input_rblock = net
            net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d1_nbl%d' % i)(net)
            net = BatchNormalization(name='bn1_nbl%d' % i)(net)

            net = Activation('relu', name='relu_nbl%d' % i)(net)

            net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d2_nbl%d' % i)(net)
            net = BatchNormalization(name='bn2_nbl%d' % i)(net)

            net = Add(name='add_nbl%d' % i)([net, input_rblock])

        net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d_output1')(net)
        net = Add(name='add_output1')([net, input_net])
        net = Conv2D(filters=num_filters, kernel_size=3, padding='same', name='conv2d_output2')(net)
        net = Conv2D(filters=1, kernel_size=9, padding='same', name='final_output')(net)

        return keras.Model(inputs=input_data, outputs=net)

    def set_parameter(self, config):
        # Saving Parameter
        self.model_path = config['PAR_EDSR']['model_path']
        self.validation_path = config['PAR_EDSR']['validation_path']
        self.config_path = config['PAR_EDSR']['config_path']

        # Training Parament
        self.batch_size = int(config['PAR_EDSR']['batch_size'])
        self.epochs = int(config['PAR_EDSR']['epochs'])
        self.epoch_save_model = int(config['PAR_EDSR']['epoch_save_model'])
        self.epoch_save_valid = int(config['PAR_EDSR']['epoch_save_valid'])
        self.learning_rate = float(config['PAR_EDSR']['learning_rate'])

        # Make Sure the path is empty
        if os.path.exists(self.validation_path):
            shutil.rmtree(self.validation_path, ignore_errors=True)
        os.makedirs(self.validation_path)

        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path, ignore_errors=True)
        os.makedirs(self.model_path)

        shutil.copy('./config.ini', self.config_path)

