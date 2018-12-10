from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPool2D, concatenate, UpSampling2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import shutil
import tensorflow as tf
from util import imgProcess, dataRecorder


class KesUNet(object):

    def __init__(self, config):

        self.set_parameter(config)

        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, 1)

        self.net = self.build_network([320, 320, 1],  self.rate_dropout, self.root_filters, self.kernel_size, self.level)
        self.net.summary()
        self.net.compile(optimizer=Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error,
                         metrics=[psnr])

    def set_parameter(self, config):
        # Saving Parameter
        self.model_path = config['PAR_UNET']['model_path']
        self.validation_path = config['PAR_UNET']['validation_path']
        self.config_path = config['PAR_UNET']['config_path']

        # Training Parament
        self.batch_size = int(config['PAR_UNET']['batch_size'])
        self.epochs = int(config['PAR_UNET']['epochs'])
        self.epoch_save_model = int(config['PAR_UNET']['epoch_save_model'])
        self.epoch_save_valid = int(config['PAR_UNET']['epoch_save_valid'])
        self.learning_rate = float(config['PAR_UNET']['learning_rate'])

        # Network Parameter
        self.rate_dropout = float(config['PAR_UNET']['rate_dropout'])
        self.root_filters = int(config['PAR_UNET']['root_filters'])
        self.kernel_size = int(config['PAR_UNET']['kernel_size'])
        self.level = int(config['PAR_UNET']['level'])

    def train(self, data, valid):

        # Make Sure the path is empty
        if os.path.exists(self.validation_path):
            shutil.rmtree(self.validation_path, ignore_errors=True)
        os.makedirs(self.validation_path)

        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path, ignore_errors=True)
        os.makedirs(self.model_path)

        shutil.copy('./config.ini', self.config_path)

        validation_path = self.validation_path
        save_epoch_valid = self.epoch_save_valid

        class SaveValidation(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.visplot.add_epoch(logs['loss'], logs['val_loss'], logs['psnr'], logs['val_psnr'])
                y_pre = self.model.predict(valid[0])

                psnr_valid = imgProcess.psnr(y_pre, valid[1])
                psnr_strings = str(epoch + 1) + ': '
                for i in range(psnr_valid.shape[0]):
                    psnr_strings = psnr_strings + ('%.3f ' % psnr_valid[i])

                psnr_strings = psnr_strings + ' \n'
                self.psnr_flie.write(psnr_strings)
                self.psnr_flie.close()
                self.psnr_flie = open(validation_path + 'psnrs.txt', 'a')

                if (epoch + 1) % save_epoch_valid == 0:
                    self.visplot.save(tofile=True, path=validation_path)
                    self.visimg.save(y_pre, epoch+1, tofile=True, path=validation_path)
                pass

            def on_batch_end(self, batch, logs=None):
                self.visplot.add_batches(logs['loss'], logs['psnr'])
                pass

            def on_train_begin(self, logs=None):
                self.visplot = dataRecorder.VisualPlot()
                self.visimg = dataRecorder.VisualImage(valid[0], valid[1])

                self.psnr_flie = open(validation_path + 'psnrs.txt', 'w')
                psnr_valid = imgProcess.psnr(valid[0], valid[1])
                psnr_strings = 'Initial: '
                for i in range(psnr_valid.shape[0]):
                    psnr_strings = psnr_strings + ('%.3f ' % psnr_valid[i])

                psnr_strings = psnr_strings + ' \n'
                self.psnr_flie.write(psnr_strings)
                self.psnr_flie.close()
                self.psnr_flie = open(validation_path + 'psnrs.txt', 'a')
                pass

            def on_train_end(self, logs=None):
                self.psnr_flie.close()
                pass

        # Train Network
        self.net.fit(x=data[0], y=data[1], batch_size=self.batch_size, validation_data=(valid[0], valid[1]),
                     epochs=self.epochs,
                     callbacks=[ModelCheckpoint(filepath=self.model_path + 'model_{epoch:02d}.h5',
                                                period=self.epoch_save_model, save_weights_only=True),
                                SaveValidation(),
                                ])

    def predict(self, data, path_weight):

        self.net.load_weights(path_weight)
        return self.net.predict(data)

    def build_network(self, shape_input, rate_dropout=0.1, root_filters=32, kernel_size=3, level=4):

        inputs = Input(shape=shape_input)
        conv = []

        for i in range(level):
            if i == 0:
                net = Conv2D(filters=root_filters * (2**i), kernel_size=kernel_size, padding='same',
                             activation='relu')(inputs)
            else:
                net = Conv2D(filters=root_filters * (2**i), kernel_size=kernel_size, padding='same',
                             activation='relu')(net)

            net = BatchNormalization()(net)
            net = Dropout(rate_dropout)(net)
            net = Conv2D(filters=root_filters * (2**i), kernel_size=kernel_size, padding='same', activation='relu')(net)
            net = BatchNormalization()(net)
            conv.append(net)
            net = MaxPool2D(pool_size=(2, 2))(net)

        net = Conv2D(filters=root_filters * (2**level), kernel_size=kernel_size, padding='same', activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dropout(rate_dropout)(net)
        net = Conv2D(filters=root_filters * (2**level), kernel_size=kernel_size, padding='same', activation='relu')(net)
        net = BatchNormalization()(net)

        for i in range(level):
            net = Conv2DTranspose(filters=root_filters * (2**(level-1-i)), kernel_size=kernel_size, strides=2,
                                  padding='same',
                                  activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dropout(rate_dropout)(net)

            net = concatenate([net, conv[level-1-i]], axis=-1)
            net = Conv2D(filters=root_filters * (2**(level-1-i)), kernel_size=kernel_size, padding='same',
                         activation='relu')(net)

            net = BatchNormalization()(net)
            net = Dropout(rate_dropout)(net)
            net = Conv2D(filters=root_filters * (2**(level-1-i)), kernel_size=kernel_size, padding='same',
                         activation='relu')(net)
            net = BatchNormalization()(net)

        net = Conv2D(filters=1, kernel_size=kernel_size, padding='same')(net)

        return keras.Model(inputs=inputs, outputs=net)
