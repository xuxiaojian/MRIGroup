import tensorflow as tf
import numpy as np
from datasets.base import DatasetBase
import tensorflow.keras.backend as k
import logging
from tensorboardX import SummaryWriter
import configparser
from methods import utilities
import scipy.io as sio
import datetime
from skimage.measure import compare_ssim, compare_psnr
import PIL
import locale


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

        if tf.executing_eagerly():
            self.train_x, self.train_y = train_dataset.tf_sample.make_one_shot_iterator().get_next()
            self.valid_x, self.valid_y = valid_dataset.tf_sample.make_one_shot_iterator().get_next()

            self.train_x = self.train_x.numpy(); self.train_y = self.train_y.numpy()
            self.valid_x = self.valid_x.numpy(); self.valid_y = self.valid_y.numpy()
        else:
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


class TFNetBase(object):
    # Need to be re-written in all case
    def get_output(self):
        return self.x

    def save_test(self, metrics, outputs, dataset, save_path):
        pass

    # Need to be re-written in some cases
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

    # __init__ need to be called at the end in __init__ method of child class
    def __init__(self, config: configparser.ConfigParser, config_section: str):
        self.config = config
        self.config_section = config_section

        self.input_shape = tuple(np.fromstring(self.config[self.config_section]['input_shape'], dtype=np.int, sep=',').tolist())
        self.output_shape = tuple(np.fromstring(self.config[self.config_section]['output_shape'], dtype=np.int, sep=',').tolist())

        self.dataset_iter = tf.data.Iterator.from_structure(output_types=(tf.float32, tf.float32),
                                                            output_shapes=((None, ) + self.input_shape, (None, ) + self.output_shape))
        self.x, self.y = self.dataset_iter.get_next()

        self.output = self.get_output()
        self.loss = self.get_loss()
        self.metrics, self.metrics_name = self.get_metrics()

        self.parameters_num = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        locale.setlocale(locale.LC_ALL, 'en_US')
        logging.info('Model Parameters Number: ' + locale.format("%d", self.parameters_num, grouping=True))

    # Basically, don't change the following codes.
    def test(self, test_dataset: DatasetBase):
        load_path = self.config['Setting']['experiment_folder'] + self.config['Setting']['test_folder'] + '/model/' + self.config['Test']['model_path'] + '/'
        batch_size = int(self.config['Test']['batch_size'])

        save_path = self.config['Setting']['experiment_folder'] + self.config['Setting']['test_folder'] + \
            '/test - ' + self.config['Test']['model_path'] + ' - ' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + '/'
        self.new_test_folders(save_path)
        logging.info('Model Parameters Number: ' + locale.format("%d", self.parameters_num, grouping=True))

        with tf.Session() as sess:
            self.load(sess, load_path)

            sess.run(self.dataset_iter.make_initializer(test_dataset.tf_dataset.batch(batch_size=batch_size)))
            test_iter = test_dataset.dataset_len() // batch_size
            test_metrics = np.zeros(shape=(len(self.metrics_name),))
            test_outputs = []
            for now_iter in range(test_iter):
                outputs, metrics = sess.run([self.output, self.metrics], feed_dict={k.learning_phase(): 0})
                logging.root.info('[Iter %d (%d in Total)] Test Batch-Output: ' % (now_iter, test_iter) + str(metrics))

                test_metrics += metrics
                test_outputs.append(outputs)
            test_metrics /= test_iter

        self.save_test(test_metrics, test_outputs, test_dataset, save_path)

    def train(self, train_dataset: DatasetBase, valid_dataset: DatasetBase = None):
        batch_size = int(self.config['Train']['batch_size'])
        learning_rate = float(self.config['Train']['learning_rate'])
        train_epoch = int(self.config['Train']['train_epoch'])
        save_epoch = int(self.config['Train']['save_epoch'])

        root_path = self.config['Setting']['experiment_folder'] + self.config['Setting']['train_folder'] + '/'
        model_path, validation_path = self.new_train_folders(code_path=self.config['Setting']['code_folder'], root_path=root_path)
        callback = TensorboardXCallback(config=self.config, output_path=root_path, validation_path=validation_path,
                                        train_dataset=train_dataset, valid_dataset=valid_dataset)

        train_op = self.get_train_op(learning_rate=learning_rate)
        save_compared_metrics = np.zeros(shape=(len(self.metrics_name), ))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
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

    @staticmethod
    def new_train_folders(root_path, code_path):
        utilities.copytree_code(src_path=code_path, dst_path=root_path)
        utilities.set_logging(target_path=root_path)

        model_path = root_path + 'model/'
        validation_path = root_path + 'validation/'
        utilities.new_folder(model_path)
        utilities.new_folder(validation_path)

        return model_path, validation_path

    @staticmethod
    def new_test_folders(root_path):
        utilities.new_folder(root_path)
        utilities.set_logging(target_path=root_path)

    @staticmethod
    def write_test(x, y, predict, write_path):
        def save_tiff(input_: np.array, path):
            imgs_list = []
            for i in range(input_.shape[0]):
                img_current = np.squeeze(input_[i])
                img_current -= np.amin(img_current)
                img_current /= np.amax(img_current)

                # noinspection PyUnresolvedReferences
                imgs_list.append(PIL.Image.fromarray(img_current))

            imgs_list[0].save(path, save_all=True, append_images=imgs_list[1:])

        psnr = []
        ssim = []

        for i in range(predict.shape[0]):
            psnr.append(compare_psnr(y[i], predict[i], data_range=1))
            ssim.append(compare_ssim(y[i], predict[i], data_range=1))

        psnr = np.array(psnr)
        ssim = np.array(ssim)
        index = np.arange(1, psnr.shape[0] + 1)

        data_csv = np.zeros(shape=[psnr.shape[0], 3])
        data_csv[:, 0] = index
        data_csv[:, 1] = psnr
        data_csv[:, 2] = ssim

        logging.root.info('Metrics: ' + str(data_csv.mean(0)))
        np.savetxt(write_path + '.csv', data_csv, fmt='%.4f', delimiter=',', header='index, psnr, ssim')

        save_tiff(predict, write_path + '_predict.tiff')
        save_tiff(x, write_path + '_x.tiff')
        save_tiff(y, write_path + '_y.tiff')

        mat_dict = {
            'predict': predict, 'x': x, 'y': y
        }
        sio.savemat(write_path + '.mat', mat_dict)
