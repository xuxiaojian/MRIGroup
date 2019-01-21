import configparser
import os
from util import dataLoader
import numpy as np

config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']
print('GPU Index: [ ' + config['GLOBAL']['index_gpu'] + ' ]')

dataset_path = config['GLOBAL']['dataset_path']
dataset_type = config['GLOBAL']['dataset_type']

network = config['GLOBAL']['network']
method = config['GLOBAL']['method']

if method == 'train':

    index_train_mat = np.fromstring(config['GLOBAL']['index_train_mat'], dtype=np.int, sep=',').tolist()
    index_train_imgs = np.fromstring(config['GLOBAL']['index_train_imgs'], dtype=np.int, sep=',').tolist()

    index_valid_mat = np.fromstring(config['GLOBAL']['index_valid_mat'], dtype=np.int, sep=',').tolist()
    index_valid_imgs = np.fromstring(config['GLOBAL']['index_valid_imgs'], dtype=np.int, sep=',').tolist()

    x_train, y_train, x_val, y_val, x_train_imgs, y_train_imgs, x_val_imgs, y_val_imgs = dataLoader.trainNvalid(
        dataset_path, dataset_type, index_train_mat, index_train_imgs, index_valid_mat, index_valid_imgs)

    if network == 'unet':

        from scadec import image_util, unet_bn, train

        data_provider = image_util.SimpleDataProvider(x_train, y_train)
        valid_provider = image_util.SimpleDataProvider(x_val, y_val)

        data_channels = 1
        truth_channels = 1

        ####################################################
        #                     NETWORK                      #
        ####################################################

        """
            here we specify the neural network.
        """

        # -- Network Setup -- #
        # set up args for the unet
        kwargs = {
            "layers": 5,           # how many resolution levels we want to have
            "conv_times": 2,       # how many times we want to convolve in each level
            "features_root": 64,
            # how many feature_maps we want to have as root (the following levels will calculate the
            # feature_map by multiply by 2, exp, 64, 128, 256)
            "filter_size": 3,      # filter size used in convolution
            "pool_size": 2,        # pooling size used in max-pooling
            "summaries": True
        }

        net = unet_bn.Unet_bn(img_channels=data_channels, truth_channels=truth_channels,
                              cost="mean_squared_error", **kwargs)

        batch_size = int(config['UNET']['batch_size'])  # batch size for training
        valid_size = int(config['UNET']['valid_size'])  # batch size for validating
        optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

        # output paths for results
        output_path = config['UNET']['save_path'] + '/models'
        prediction_path = config['UNET']['save_path'] + '/validation'
        # restore_path = 'gpu001/models/50099_cpkt'

        # optional args
        opt_kwargs = {
            'learning_rate': float(config['UNET']['learning_rate'])
        }

        # make a trainer for scadec
        trainer = train.Trainer_bn(net, batch_size=batch_size, optimizer="adam", opt_kwargs=opt_kwargs)
        path = trainer.train(data_provider, output_path, valid_provider, valid_size,
                             x_train_imgs, y_train_imgs, x_val_imgs, y_val_imgs,
                             training_iters=int(config['UNET']['training_iters']),
                             epochs=int(config['UNET']['epochs']),
                             save_epoch=int(config['UNET']['save_epoch']),
                             dropout=float(config['UNET']['dropout']),
                             prediction_path=prediction_path,
                             display_step=1)
