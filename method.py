from tensorflow.python.keras.layers import Conv3D, ReLU, Dropout, Conv3DTranspose, Input, MaxPool3D, \
    Concatenate, Add, Activation, UpSampling3D, multiply, Lambda
from tensorflow.python.keras.models import Model
import tensorflow as tf
from tensorboardX import SummaryWriter
import numpy as np
from tensorflow.python.keras import backend as K


def get_gating_signal(input_, output_size_):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    """
    output_ = Conv3D(output_size_, kernel_size=1, padding='same')(input_)
    output_ = ReLU()(output_)
    return output_


def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=-1), arguments={'repnum': rep})(tensor)


def get_attention_block(x, gating, inter_filter):
    # Suppose: gating: m by m x: 2m by 2m
    x_input = x
    x = Conv3D(inter_filter, kernel_size=2, strides=(1, 2, 2), padding='same')(x)  # m by m

    gating = Conv3D(inter_filter, kernel_size=1, padding='same')(gating) # m by m
    gating = Conv3DTranspose(inter_filter, kernel_size=3, strides=(1, 1, 1), padding='same')(gating)  # m by m

    x_gating = Add()([gating, x]) # m by m
    x_gating = ReLU()(x_gating)
    x_gating = Conv3D(1, kernel_size=1, padding='same')(x_gating)

    x_gating = Activation('sigmoid')(x_gating)
    x_gating = UpSampling3D((1, 2, 2))(x_gating)  # 2m by 2m
    x_gating = expend_as(x_gating, inter_filter)

    y = multiply([x_gating, x_input])

    result = Conv3D(inter_filter, kernel_size=1, padding='same')(y)

    return result


def attention_unet3d(input_shape):

    def conv3d_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv3D(filters=filters_, kernel_size=kernel_size_, padding='same')(input_)
        output_ = ReLU()(output_)
        output_ = Dropout(rate=0.1)(output_)
        return output_

    def conv3d_transpose_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(1, 2, 2))(input_)
        output_ = ReLU()(output_)
        output_ = Dropout(rate=0.1)(output_)
        return output_

    kernel_size = 3
    filters_root = 32
    conv_times = 3
    up_down_times = 4

    skip_layers_storage = []

    # Build Network
    net_input = Input(shape=input_shape)
    net = conv3d_relu_dropout(net_input, filters_root, kernel_size)

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size)

        skip_layers_storage.append(net)
        net = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(net)

    for layer in range(up_down_times - 1, -1, -1):
        # noinspection PyUnboundLocalVariable
        filters = 2 ** layer * filters_root
        print('filters: ', filters)
        print(skip_layers_storage[layer])

        gate_signal = get_gating_signal(net, filters)
        attention_block = get_attention_block(x=skip_layers_storage[layer], gating=gate_signal,
                                              inter_filter=filters)

        net = conv3d_transpose_relu_dropout(net, filters, kernel_size)
        net = Concatenate(axis=-1)([net, attention_block])

        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size)

    net = Conv3D(filters=1, kernel_size=1, padding='same')(net)
    # net = Add()([net, net_input])

    return Model(inputs=net_input, outputs=net)


def unet3d(input_shape):

    def conv3d_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv3D(filters=filters_, kernel_size=kernel_size_, padding='same')(input_)
        output_ = ReLU()(output_)
        output_ = Dropout(rate=0.1)(output_)
        return output_

    def conv3d_transpose_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(1, 2, 2))(input_)
        output_ = ReLU()(output_)
        output_ = Dropout(rate=0.1)(output_)
        return output_

    kernel_size = 3
    filters_root = 32
    conv_times = 3
    up_down_times = 4

    skip_layers_storage = []

    # Build Network
    net_input = Input(shape=input_shape)
    net = conv3d_relu_dropout(net_input, filters_root, kernel_size)

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size)

        skip_layers_storage.append(net)
        net = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(net)

    for layer in range(up_down_times - 1, -1, -1):

        filters = 2 ** layer * filters_root
        net = conv3d_transpose_relu_dropout(net, filters, kernel_size)
        net = Concatenate(axis=-1)([net, skip_layers_storage[layer]])

        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size)

    net = Conv3D(filters=1, kernel_size=1, padding='same')(net)
    net = Add()([net, net_input])

    return Model(inputs=net_input, outputs=net)


class KerasCallBack(tf.keras.callbacks.Callback):
    def __init__(self, output_path, train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset):
        super().__init__()
        self.writer = SummaryWriter(output_path)
        self.output_path = output_path
        self.global_batch = 0

        with tf.Session() as sess:
            self.train_x, self.train_y = sess.run(train_dataset.make_one_shot_iterator().get_next())
            self.valid_x, self.valid_y = sess.run(valid_dataset.make_one_shot_iterator().get_next())

    def on_train_begin(self, logs=None):
        for i in range(self.train_x.shape[0]):
            self.add_imgs(self.train_x[i], tag='train/%d_x' % i, step=0)
            self.add_imgs(self.train_y[i], tag='train/%d_y' % i, step=0)

            self.add_imgs(self.valid_x[i], tag='valid/%d_x' % i, step=0)
            self.add_imgs(self.valid_y[i], tag='valid/%d_y' % i, step=0)

    def on_train_batch_end(self, batch, logs=None):
        self.writer.add_scalar(tag='train_batch/loss', scalar_value=logs['loss'], global_step=self.global_batch)
        self.writer.add_scalar(tag='train_batch/psnr', scalar_value=logs['psnr_tf'], global_step=self.global_batch)
        self.writer.add_scalar(tag='train_batch/ssim', scalar_value=logs['ssim_tf'], global_step=self.global_batch)
        self.global_batch += 1

    def on_epoch_end(self, epoch, logs=None):
        self.writer.add_scalar(tag='train_epoch/loss', scalar_value=logs['loss'], global_step=epoch)
        self.writer.add_scalar(tag='train_epoch/psnr', scalar_value=logs['psnr_tf'], global_step=epoch)
        self.writer.add_scalar(tag='train_epoch/ssim', scalar_value=logs['ssim_tf'], global_step=epoch)

        self.writer.add_scalar(tag='valid_epoch/loss', scalar_value=logs['val_loss'], global_step=epoch)
        self.writer.add_scalar(tag='valid_epoch/psnr', scalar_value=logs['val_psnr_tf'], global_step=epoch)
        self.writer.add_scalar(tag='valid_epoch/ssim', scalar_value=logs['val_ssim_tf'], global_step=epoch)

        train_pre = self.model.predict(self.train_x, batch_size=4)
        test_pre = self.model.predict(self.valid_x, batch_size=4)

        for i in range(train_pre.shape[0]):
            self.add_imgs(train_pre[i], tag='train/%d_predict' % i, step=epoch)
            self.add_imgs(test_pre[i], tag='valid/%d_predict' % i, step=epoch)

    def add_imgs(self, imgs, tag, step):
        from skimage.color import gray2rgb

        depth, width, height, channel = imgs.shape
        if channel == 1:
            imgs_rgb = np.zeros(shape=[depth, height, width, 3])
            for i in range(imgs.shape[0]):
                imgs_rgb[i, :, :, :] = gray2rgb(np.squeeze(imgs[i]))
            self.writer.add_images(tag=tag, img_tensor=imgs_rgb, global_step=step, dataformats='NHWC')

        else:
            if channel == 3:
                self.writer.add_images(tag=tag, img_tensor=imgs, global_step=step, dataformats='NHWC')

            else:
                imgs_conca = np.concatenate([imgs[:, :, :, 0], imgs[:, :, :, 1]], axis=0)
                batch_conca, width_conca, height_conca = imgs_conca.shape

                imgs_rgb = np.zeros(shape=[batch_conca, width_conca, height_conca, 3])
                for i in range(batch_conca):
                    imgs_rgb[i, :, :, :] = gray2rgb(np.squeeze(imgs_conca[i]))
                self.writer.add_images(tag=tag, img_tensor=imgs_rgb, global_step=step, dataformats='NHWC')
