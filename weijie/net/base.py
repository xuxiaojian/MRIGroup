import tensorflow as tf
from tensorflow import keras as keras
from util.data_recorder import psnr_metrics, ssim_metrics, new_folder, KearsCallback
from tensorflow.python.keras.callbacks import ModelCheckpoint


# noinspection PyNoneFunctionAssignment,PyTupleAssignmentBalance
class BaseKaresNetwork(object):

    ####################################################
    # BEGIN: FUNCTION USER MUST INHERIT
    ####################################################
    def set_test_imgs(self):
        pass

    def set_train_imgs(self):
        pass

    def set_network(self):
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

        self.network = self.set_network()
        if self.FLAGS_DICT['gpu_num'] > 1:
            self.network = keras.utils.multi_gpu_model(self.network, gpus=self.FLAGS_DICT['gpu_num'])

        self.network.summary()
        self.network.compile(optimizer=self._get_optimizer(), loss=self._get_loss(), metrics=[psnr_metrics, ssim_metrics])

    def _get_optimizer(self):
        return keras.optimizers.Adam(self.FLAGS_DICT[self.config_name + 'learning_rate'])

    def _get_loss(self):
        return keras.losses.mean_squared_error

    def train(self):
        x_train, y_train, x_train_imgs, y_train_imgs, x_val, y_val, x_val_imgs, y_val_imgs = self.set_train_imgs()

        model_path = self.FLAGS_DICT['output_path'] + 'model/'
        new_folder(self.FLAGS_DICT['output_path'])
        new_folder(model_path)

        costom_callback = KearsCallback(x_train_imgs, y_train_imgs, x_val_imgs, y_val_imgs,
                                        self.FLAGS_DICT['output_path'],
                                        tf.flags.FLAGS.flags_into_string().replace('\n', '\n\n'),
                                        self.FLAGS_DICT[self.config_name + 'epoch_save_val'])
        save_model_callback = ModelCheckpoint(filepath=model_path + '{epoch:02d}_weight.h5',
                                              period=self.FLAGS_DICT[self.config_name + 'epoch_save_model'],
                                              save_weights_only=True)

        self.network.fit(
            x=x_train, y=y_train,
            batch_size=self.FLAGS_DICT[self.config_name + 'batch_size'] * self.FLAGS_DICT['gpu_num'],
            epochs=self.FLAGS_DICT[self.config_name + 'epoch'],
            validation_data=(x_val, y_val),
            callbacks=[costom_callback, save_model_callback]
        )

        self.network.save_weights(model_path + 'final_weight.h5')
        self.network.save(model_path + 'final_model.h5')

    def predict(self, x):
        return self.network.predict(x)
