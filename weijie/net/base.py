import tensorflow as tf
from tensorflow import keras as keras
from data.recorder import new_folder, KearsCallback
from tensorflow.python.keras.callbacks import ModelCheckpoint
import shutil


# Custom metrics for Kearas
def psnr_metrics(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))


def ssim_metrics(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))


# noinspection PyNoneFunctionAssignment,PyTupleAssignmentBalance
class BaseKaresNetwork(object):

    ####################################################
    # BEGIN: FUNCTION USER MUST INHERIT
    ####################################################
    def get_train_data(self):
        pass

    def get_network(self):
        pass
    ####################################################
    # END: FUNCTION USER MUST INHERIT
    ####################################################

    def __call__(self, mode):
        if mode == 'train':
            self.train()

    def __init__(self, config_name):
        self.FLAGS_DICT = tf.flags.FLAGS.flag_values_dict()
        self.config_name = config_name

        self.network = self.get_network()
        if self.FLAGS_DICT['global_gpu_num'] > 1:
            self.network = keras.utils.multi_gpu_model(self.network, gpus=self.FLAGS_DICT['global_gpu_num'])

        self.network.summary()
        self.network.compile(optimizer=self.get_optimizer(), loss=self.get_loss(), metrics=[psnr_metrics, ssim_metrics])

    def get_optimizer(self):
        return keras.optimizers.Adam(self.FLAGS_DICT[self.config_name + 'learning_rate'])

    def get_loss(self):
        return keras.losses.mean_squared_error

    def train(self):

        new_folder(self.FLAGS_DICT['global_output_path'])
        shutil.copytree('/export/project/gan.weijie/MRIGroup/weijie/', self.FLAGS_DICT['global_output_path'] + 'code/')

        x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs = self.get_train_data()

        costom_callback = KearsCallback(x_train_imgs, y_train_imgs, x_val_imgs, y_val_imgs,
                                        self.FLAGS_DICT['global_output_path'],
                                        tf.flags.FLAGS.flags_into_string().replace('\n', '\n\n'),
                                        self.FLAGS_DICT[self.config_name + 'epoch_save'])

        self.network.fit(
            x=x_train, y=y_train,
            batch_size=self.FLAGS_DICT[self.config_name + 'batch_size'] * self.FLAGS_DICT['global_gpu_num'],
            epochs=self.FLAGS_DICT[self.config_name + 'epoch'],
            validation_data=(x_val, y_val),
            callbacks=[costom_callback]
        )