import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
import scipy.io as sio
from skimage.color import gray2rgb
from data.tools import new_folder
import logging


# noinspection PyMethodMayBeStatic
class TFBase(object):
    #################################
    # The following functions should be re-wrote
    #################################
    def get_net_output(self):
        return 0

    #################################
    # The following function may need to be re-wrote, : )
    #################################
    def get_metrics(self):
        psnr = tf.image.psnr(self.y_output, self.y_gt, 1)
        ssim = tf.image.ssim(self.y_output, self.y_gt, 1)

        metrics = [self.loss, psnr, ssim]
        metrics_name = ["LOSS", "PSNR", "SSIM"]

        return [metrics, metrics_name]
    
    def get_train_op(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def get_loss(self):
        return tf.losses.mean_squared_error(self.y_output, self.y_gt)

    #################################
    # The following function should not be changed in common.
    #################################
    def __init__(self, input_shape, output_shape):
        self.x = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self.y_gt = tf.placeholder(dtype=tf.float32, shape=output_shape)

        self.bn_training = tf.placeholder(tf.bool)  # Only for Batch Normalization Layer
        self.dropout_training = tf.placeholder(tf.bool)

        self.dropout_rate = tf.placeholder(tf.float32)  # Only for Drop Out Layer
        self.y_output = self.get_net_output()

        self.lr = tf.placeholder(tf.float32)

        self.loss = self.get_loss()
        self.train_op = self.get_train_op()

        self.metrics, self.metrics_name = self.get_metrics()

    def predict(self, x, y, batch_size, model_path):

        pre = np.zeros(shape=x.shape, dtype=np.float64)

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            TFTrainer.restore_model(sess=sess, path=model_path)

            size = x.shape[0]
            if size > 1:
                index = TFTrainer.make_batches(size=size, batch_size=batch_size)

                num_batches = index.__len__()

                nums_metrics = self.metrics_name.__len__()
                epoch_metrics = np.zeros(shape=[nums_metrics, 1])

                for batch, (batch_start, batch_end) in enumerate(index):
                    x_batch = x[batch_start:batch_end]
                    y_batch = y[batch_start:batch_end]

                    pre[batch_start:batch_end], metrics = sess.run([self.y_output, self.metrics],
                                                                   feed_dict={self.x: x_batch,
                                                                   self.y_gt: y_batch,
                                                                   self.dropout_rate: 0,
                                                                   self.bn_training: False,
                                                                   self.dropout_training: False})

                    for i in range(nums_metrics):
                        metric = metrics[i]
                        metric = metric.mean()
                        epoch_metrics[i] += metric

                    verbose_info = "[Info] Prediction Output: Batch = [%d]" % (batch + 1)
                    logging.root.info(verbose_info)

                    verbose_info = "Metrics: "
                for i in range(nums_metrics):
                    epoch_metrics[i] /= num_batches
                    verbose_info += self.metrics_name[i] + ": [%.4f]. " % epoch_metrics[i]

                logging.root.info(verbose_info)

            else:
                pre = sess.run(self.y_output, feed_dict={self.x: x, self.dropout_rate: 0,
                                                         self.bn_training: False, self.dropout_training: False})

        return pre

    @staticmethod
    def to_mat(data, name, path):
        sio.savemat(path + name + '.mat', {name: data})


class TFTrainer(object):
    def __init__(self, net, path, config_info, lr=0.01, batch_size=32, train_epoch=100, save_epoch=20, dropout_value=0):
        """

        :type net: Should have 'x', 'y_gt', 'y_output', 'matrices', 'matrices_name' and 'train_op' object members
        """
        self.save_epoch = save_epoch
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.dropout_value = dropout_value
        self.lr = lr

        self.net = net
        self.path = path
        self.config_info = config_info

    def run(self, train_x, train_y, valid_x, valid_y, train_x_imgs, train_y_imgs, valid_x_imgs, valid_y_imgs,
            batch_verbose=True, epoch_verbose=True):

        ################
        # Set up shuffle index
        ################
        nums_train = train_x.shape[0]
        nums_valid = valid_x.shape[0]

        index_batches_train = self.make_batches(nums_train, self.batch_size)
        index_train = np.arange(nums_train)

        index_batches_valid = self.make_batches(nums_valid, self.batch_size)

        num_batches_train = index_batches_train.__len__()
        num_batches_valid = index_batches_valid.__len__()

        nums_metrics = self.net.metrics_name.__len__()

        batch_verbose_times = None
        if batch_verbose:
            batch_verbose_times = int(num_batches_train / 10)

        ################
        # Initial writer
        ################
        writer = TBXWriter(self.path, self.config_info)
        writer.imgs_train_init(train_x_imgs, train_y_imgs)
        writer.imgs_valid_init(valid_x_imgs, valid_y_imgs)

        ################
        # Initial .mat save path
        ################
        imgs_save_path = self.path + "mat/"
        new_folder(imgs_save_path)
        sio.savemat(imgs_save_path + "init.mat", {"x": valid_x_imgs, "y": valid_y_imgs})

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())

            writer_step = 0
            for epoch in range(self.train_epoch):

                ################
                # Training
                ################
                np.random.shuffle(index_train)
                epoch_metrics = np.zeros(shape=[nums_metrics, 1])
                for batch, (batch_start, batch_end) in enumerate(index_batches_train):
                    batch_ids = index_train[batch_start:batch_end]
                    x = train_x[batch_ids]
                    y_gt = train_y[batch_ids]

                    ################
                    # Run
                    ################
                    _, metrics = sess.run([self.net.train_op, self.net.metrics],
                                          feed_dict={self.net.x: x, self.net.y_gt: y_gt,
                                                     self.net.dropout_rate: self.dropout_value,
                                                     self.net.lr: self.lr,
                                                     self.net.bn_training: True,
                                                     self.net.dropout_training: True})

                    ################
                    # Verbose
                    ################
                    verbose_info = "[Info] Batch Output: Batch = [%d] Totally [%d]. " \
                                   % (batch + 1, num_batches_train)

                    for i in range(nums_metrics):
                        metric = metrics[i]
                        metric = metric.mean()

                        epoch_metrics[i] += metric
                        verbose_info += self.net.metrics_name[i] + ": [%.4f]. " % metric

                    if batch_verbose:
                        if (batch + 1) % batch_verbose_times == 0:
                            logging.root.info(verbose_info)

                    ################
                    # Add data into writer
                    ################
                    writer.train_batch(metrics, self.net.metrics_name, writer_step)
                    writer_step += 1

                verbose_info = "[Info] Epoch Output: Epoch = [%d] Totally [%d]. " % (epoch + 1, self.train_epoch)
                for i in range(nums_metrics):
                    epoch_metrics[i] /= num_batches_train
                    verbose_info += self.net.metrics_name[i] + ": [%.4f]. " % epoch_metrics[i]

                if epoch_verbose:
                    logging.root.info(verbose_info)

                ################
                # Add data into writer
                ################
                writer.train_epoch(epoch_metrics, self.net.metrics_name, writer_step)
                epoch_imgs = sess.run(self.net.y_output, feed_dict={self.net.x: train_x_imgs,
                                                                    self.net.dropout_rate: 0,
                                                                    self.net.bn_training: False,
                                                                    self.net.dropout_training: False})
                writer.imgs_train_epoch(epoch_imgs, writer_step)

                ################
                # Validation
                ################
                epoch_metrics = np.zeros(shape=[nums_metrics, 1])
                for batch, (batch_start, batch_end) in enumerate(index_batches_valid):
                    x = valid_x[batch_start: batch_end]
                    y_gt = valid_y[batch_start: batch_end]

                    metrics = sess.run(self.net.metrics, feed_dict={self.net.x: x,
                                                                    self.net.y_gt: y_gt,
                                                                    self.net.dropout_rate: 0,
                                                                    self.net.bn_training: False,
                                                                    self.net.dropout_training: False})

                    for i in range(nums_metrics):
                        metric = metrics[i]
                        metric = metric.mean()
                        epoch_metrics[i] += metric

                ################
                # Verbose
                ################
                verbose_info = "[Info] Validation Output: Epoch = [%d] Totally [%d]. " % (epoch + 1, self.train_epoch)
                for i in range(nums_metrics):
                    epoch_metrics[i] /= num_batches_valid
                    verbose_info += self.net.metrics_name[i] + ": [%.4f]. " % epoch_metrics[i]

                if epoch_verbose:
                    logging.root.info(verbose_info)

                ################
                # Add data into writer
                ################
                writer.valid_epoch(epoch_metrics, self.net.metrics_name, writer_step)
                epoch_imgs = sess.run(self.net.y_output, feed_dict={self.net.x: valid_x_imgs,
                                                                    self.net.dropout_rate: 0,
                                                                    self.net.bn_training: False,
                                                                    self.net.dropout_training: False})
                writer.imgs_valid_epoch(epoch_imgs, writer_step)
                ################
                # Save .mat
                ################
                sio.savemat(imgs_save_path + "epoch_%d.mat" % (epoch + 1), {"pre": epoch_imgs})

                ################
                # Save Model
                ################
                if (epoch + 1) % self.save_epoch == 0:
                    model_save_path = self.path + "model/epoch_" + str(epoch + 1) + "/"
                    new_folder(model_save_path)
                    self.save_model(sess, model_save_path)

    @staticmethod
    def make_batches(size, batch_size):
        num_batches = (size + batch_size - 1) // batch_size  # round up
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(num_batches)]

    @staticmethod
    def save_model(sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path + 'model.ckpt')

    @staticmethod
    def restore_model(sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path + 'model.ckpt')


# Convert a python dict to markdown table
def config_to_markdown_table(config_dict, name_section):
    info = '## ' + name_section + '\n'
    info = info + '|  Key  |  Value |\n|:----:|:---:|\n'

    for i in config_dict.keys():
        info = info + '|' + i + '|' + config_dict[i] + '|\n'

    info = info + '\n'

    return info


# A custom TensorboardX Class
class TBXWriter(object):
    def __init__(self, path, config_info):
        self.writer = SummaryWriter(path)
        self.writer.add_text(tag='config', text_string=config_info, global_step=0)

    ###############
    # Metrics
    ###############

    def train_batch(self, metrics, metrics_name, step):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='train_batch/' + metrics_name[i],
                                   scalar_value=metrics[i].mean(), global_step=step)

    def train_epoch(self, metrics, metrics_name, step):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='train_epoch/' + metrics_name[i],
                                   scalar_value=metrics[i].mean(), global_step=step)

    def valid_epoch(self, metrics, metrics_name, step):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='valid_epoch/' + metrics_name[i],
                                   scalar_value=metrics[i].mean(), global_step=step)

    ###############
    # Image
    ###############
    @staticmethod
    def img_preprocess(imgs):
        imgs -= np.amin(imgs)
        imgs /= np.amax(imgs)
        return imgs

    def show_img(self, imgs, tag, step):
        shape = imgs.shape.__len__()

        def hw():
            self.writer.add_image(tag=tag, img_tensor=imgs, global_step=step, dataformats='HW')

        def hwc():
            if imgs.shape[2] == 1:
                img_cur = (np.squeeze(imgs))
                self.writer.add_image(tag=tag, img_tensor=img_cur, global_step=step, dataformats='HW')
            else:
                self.writer.add_image(tag=tag, img_tensor=imgs, global_step=step, dataformats='HWC')

        def nhwc():
            if imgs.shape[3] == 1:
                img_cur = np.zeros(shape=[imgs.shape[0], imgs.shape[1], imgs.shape[2], 3])

                for i in range(imgs.shape[0]):
                    img_cur[i, :, :, :] = gray2rgb(np.squeeze(imgs[i]))

                self.writer.add_images(tag=tag, img_tensor=img_cur, global_step=step, dataformats='NHWC')
            else:

                if imgs.shape[3] == 2:
                    imgs_ = np.concatenate([imgs[:, :, :, 0], imgs[:, :, :, 1]], axis=0)

                    img_cur = np.zeros(shape=[imgs_.shape[0], imgs_.shape[1], imgs_.shape[2], 3])

                    for i in range(imgs_.shape[0]):
                        img_cur[i, :, :, :] = gray2rgb(np.squeeze(imgs_[i]))

                    self.writer.add_images(tag=tag, img_tensor=img_cur, global_step=step, dataformats='NHWC')

                else:
                    self.writer.add_images(tag=tag, img_tensor=imgs, global_step=step, dataformats='NHWC')

        shape_dict = {
            2: hw,
            3: hwc,
            4: nhwc,
        }

        shape_dict[shape]()

    def imgs_train_epoch(self, imgs, step):
        for i in range(imgs.shape[0]):
            self.show_img(imgs=imgs[i], tag='train/index%d_pre' % i, step=step)

    def imgs_train_init(self, x_imgs, y_imgs):
        for i in range(x_imgs.shape[0]):
            self.show_img(imgs=x_imgs[i], tag='train/index%d_data' % i, step=0)
            self.show_img(imgs=y_imgs[i], tag='train/index%d_label' % i, step=0)

    def imgs_valid_epoch(self, imgs, step):
        for i in range(imgs.shape[0]):
            self.show_img(imgs=imgs[i], tag='valid/index%d_pre' % i, step=step)

    def imgs_valid_init(self, x_imgs, y_imgs):
        for i in range(x_imgs.shape[0]):
            self.show_img(imgs=x_imgs[i], tag='valid/index%d_data' % i, step=0)
            self.show_img(imgs=y_imgs[i], tag='valid/index%d_label' % i, step=0)
