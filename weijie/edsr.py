import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Add, PReLU
import shutil
import tensorflow as tf


class KesNetwork(object):

    def __init__(self, shape_input, num_rblocks):

        self.net = self.layers(shape_input=shape_input, num_rblocks=num_rblocks)

        # Define Custom metrics: PSNR
        def psnr(y_true, y_pred):
            return tf.image.psnr(y_pred, y_true, max_val=1)

        self.net.compile(optimizer='adam', loss=keras.losses.mean_squared_error, metrics=[psnr])
        self.net.summary()

    def layers(self, shape_input, num_rblocks, num_filters=32):

        input_data = Input(shape=shape_input)
        net = Conv2D(filters=64, kernel_size=9, padding='same')(input_data)
        # kernel_size = 9 is learnt from SRResNet
        input_net = net

        for i in range(num_rblocks):
            input_rblock = net
            net = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(net)
            net = PReLU()(net)
            # kernel_size = 9 is learnt from SRResNet
            net = Conv2D(filters=64, kernel_size=3, padding='same')(net)
            net = Add()([net, input_rblock])

        net = Conv2D(filters=64, kernel_size=3, padding='same')(net)
        net = Add()([net, input_net])
        net = Conv2D(filters=64, kernel_size=3, padding='same')(net)
        net = Conv2D(filters=1, kernel_size=9, padding='same')(net)

        return keras.Model(inputs=input_data, outputs=net)

    def train(self, data_provider, valid_provider,
              validation_path, output_path,
              batch_size, epochs, save_epoch):

        if os.path.exists(validation_path):
            shutil.rmtree(validation_path, ignore_errors=True)
        os.makedirs(validation_path)

        if os.path.exists(output_path):
            shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)

        self.net.fit(x=data_provider[0], y=data_provider[1], batch_size=batch_size, epochs=epochs,
                     validation_data=(valid_provider[0], valid_provider[1]),
                     callbacks=[keras.callbacks.TensorBoard(log_dir=output_path),  # Record Training Data in TFB
                                keras.callbacks.ModelCheckpoint(filepath=output_path + 'model_{epoch:02d}.h5',
                                                                period=save_epoch),  # Svae Model
                                ])

        self.net.save(filepath=output_path + 'final.h5')


if __name__ == '__main__':
    from util import data_loader
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    network = KesNetwork(shape_input=[28, 28, 1], num_rblocks=16)
    train_imgs, test_imgs = data_loader.read_mnist(50000, 50)
    x_train, y_train = train_imgs(50000, fix=True)
    x_test, y_test = test_imgs(50, fix=True)

    network.train(data_provider=[x_train, y_train], valid_provider=[x_test, y_test],
                  validation_path='/home/xiaojianxu/gan/MRIGroup/weijie/result/edsr_mnist/validation/',
                  output_path='/home/xiaojianxu/gan/MRIGroup/weijie/result/edsr_mnist/model/',
                  batch_size=32, epochs=500, save_epoch=100)
