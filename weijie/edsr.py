import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Add, PReLU, Activation
from util import data_recorder
import shutil


class KesNetwork(object):

    def __init__(self, shape_input, num_rblocks):

        self.net = self.layers(shape_input=shape_input, num_rblocks=num_rblocks)
        self.net.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

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
              batch_size, epochs, save_epoch, valid_epoch):

        if os.path.exists(validation_path):
            shutil.rmtree(validation_path, ignore_errors=True)
        os.makedirs(validation_path)

        if os.path.exists(output_path):
            shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)

        data_recorder.save_mat(valid_provider[0], "%s%s.mat" % (validation_path, 'origin_x'))
        data_recorder.save_mat(valid_provider[1], "%s%s.mat" % (validation_path, 'origin_y'))

        for i in range(epochs):
            print("[edsr Training]: epoch - ", i)
            self.net.fit(x=data_provider[0], y=data_provider[1], batch_size=batch_size, epochs=1,
                         callbacks=[keras.callbacks.TensorBoard(log_dir=output_path)])

            if (i + 1) % save_epoch == 0:
                self.net.save(output_path + "model_%s.h5" % (i + 1))

            if (i + 1) % valid_epoch == 0:
                hat_y = self.net.predict(x=valid_provider[0])
                data_recorder.save_mat(hat_y, validation_path + "%s.mat" % (i + 1))


if __name__ == '__main__':
    from util import data_loader
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    network = KesNetwork(shape_input=[28, 28, 1], num_rblocks=16)
    # train_imgs, test_imgs = data_loader.read_mnist(3000, 5)
    # x_train, y_train = train_imgs(3000, fix=True)
    # x_test, y_test = test_imgs(5, fix=True)
    #
    # network.train(data_provider=[x_train, y_train], valid_provider=[x_test, y_test],
    #               validation_path='/home/xiaojianxu/gan/MRIGroup/weijie/result/edsr_mnist/validation/',
    #               output_path='/home/xiaojianxu/gan/MRIGroup/weijie/result/edsr_mnist/model/',
    #               batch_size=32, epochs=200, save_epoch=50, valid_epoch=20)

