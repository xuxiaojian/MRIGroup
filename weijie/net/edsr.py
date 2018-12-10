import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Add, PReLU, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import shutil
import tensorflow as tf
import scipy.io as sio
from scipy import misc
import numpy as np
from util.dataRecorder import VisualinPyplot
import os


class KesNetwork(object):

    def __init__(self, shape_input, shape_test, num_rblocks=16, num_filters=64):

        self.net = self.layers(shape_input=shape_input, num_rblocks=num_rblocks, num_filters=num_filters)
        self.test_net = self.layers(shape_input=shape_test, num_rblocks=num_rblocks, num_filters=num_filters)
        self.shape_test = shape_test

        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, max_val=1)

        self.net.compile(optimizer=Adam(lr=0.0001), loss=keras.losses.mean_squared_error, metrics=[psnr])
        self.test_net.compile(optimizer=Adam(lr=0.0001), loss=keras.losses.mean_squared_error, metrics=[psnr])

        self.net.summary()

    def layers(self, shape_input, num_rblocks=16, num_filters=64):

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

    def train(self, data_provider, valid_provider, test_provider,
              test_path, output_path,
              batch_size, epochs, epoch_save_model, epoch_save_testmat):

        # Make Sure the path is empty
        if os.path.exists(test_path):
            shutil.rmtree(test_path, ignore_errors=True)
        os.makedirs(test_path)

        if os.path.exists(output_path):
            shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)

        # Visualization Function
        epoch_loss_train = VisualinPyplot('loss_trian', output_path, index_fig=1)
        epoch_loss_valid = VisualinPyplot('loss_valid', output_path, index_fig=2)
        epoch_loss_test = VisualinPyplot('loss_test', test_path, index_fig=3)

        epoch_psnr_train = VisualinPyplot('psnr_train', output_path, index_fig=4)
        epoch_psnr_valid = VisualinPyplot('psnr_valid', output_path, index_fig=5)
        epoch_psnr_test = VisualinPyplot('psnr_test', test_path, index_fig=6)

        def visual(epoch_loss_train_cur, epoch_loss_valid_cur, epoch_loss_test_cur, epoch_psnr_train_cur,
                   epoch_psnr_valid_cur, epoch_psnr_test_cur):
            epoch_loss_train.add(epoch_loss_train_cur)
            epoch_loss_valid.add(epoch_loss_valid_cur)
            epoch_loss_test.add(epoch_loss_test_cur)
            epoch_psnr_train.add(epoch_psnr_train_cur)
            epoch_psnr_valid.add(epoch_psnr_valid_cur)
            epoch_psnr_test.add(epoch_psnr_test_cur)

        # Epoch Callback
        test_net = self.test_net
        shape_test = self.shape_test[0]

        num_predict = test_provider[0].shape[0]
        for i in range(num_predict):
            os.makedirs(test_path + '%s/' % i)

        class SaveValidation(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Test Network with different size Input
                test_net.set_weights(self.model.get_weights())

                # Obtain Test Data
                y_predcit = test_net.predict(test_provider[0])
                loss_test, psnr_test = test_net.evaluate(x=test_provider[0], y=test_provider[1])

                if (epoch+1) % epoch_save_testmat == 0:

                    sio.savemat(test_path + '%d.mat' % (epoch+1), {
                        'y_pre': y_predcit,
                        'x_sr': test_provider[0],
                        'y': test_provider[1],
                        'x': test_provider[2],
                    })

                for i in range(num_predict):
                    test = np.zeros(shape=[shape_test, shape_test * 4, 1])

                    test[:, :shape_test, :] = test_provider[2][i, :, :, :]
                    test[:, shape_test:shape_test * 2, :] = test_provider[0][i, :, :, :]
                    test[:, shape_test * 2:shape_test * 3, :] = y_predcit[i, :, :, :]
                    test[:, shape_test * 3:shape_test * 4, :] = test_provider[1][i, :, :, :]
                    test.shape = [shape_test, shape_test * 4]

                    misc.imsave(test_path + '%d/%d.png' % (i,epoch+1), test)

                # Write All epoch data into png file
                visual(logs['loss'], logs['val_loss'], loss_test, logs['psnr'], logs['val_psnr'], psnr_test)

        self.net.fit(x=data_provider[0], y=data_provider[1], validation_data=(valid_provider[0], valid_provider[1]),
                     batch_size=batch_size, epochs=epochs,
                     callbacks=[keras.callbacks.ModelCheckpoint(filepath=output_path + 'weight_only_{epoch:02d}.h5',
                                                                period=epoch_save_model, save_weights_only=True),
                                SaveValidation(),
                                ])

        # Save Final model
        self.net.save_weights(filepath=output_path + 'weight_only_final.h5')
        self.net.save(filepath=output_path + 'model_final.h5')


if __name__ == '__main__':

    print('edsr.py')

    # from util import data_loader
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    #
    # network = KesNetwork(shape_input=[28, 28, 1], shape_test=[56, 56, 1], num_rblocks=2, num_filters=3)
    # train_imgs, valid_imgs, test_imgs = data_loader.read_mnist(num_train=10000, num_vaild=50, num_test=5, shape_test=56)
    # x_train, y_train = train_imgs(10000, fix=True)
    # x_valid, y_valid = valid_imgs(50, fix=True)
    # x_test, y_test = test_imgs(5, fix=True)
    #
    # network.train(data_provider=[x_train, y_train], valid_provider=[x_valid, y_valid], test_provider=[x_test, y_test],
    #               test_path='/home/xiaojianxu/gan/MRIGroup/weijie/result/edsr_mnist/test/',
    #               output_path='/home/xiaojianxu/gan/MRIGroup/weijie/result/edsr_mnist/model/',
    #               batch_size=32, epochs=100, epoch_save_model=20, epoch_save_testmat=5)
