import configparser
import os
import sys
from net import unet, sr
from util import dataLoader
import numpy as np

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']
print('Using [', np.fromstring(config['GLOBAL']['index_gpu'], dtype=int, sep=',').shape[0],
      '] GPUs. The Index: [ ', config['GLOBAL']['index_gpu'], ' ]')

dataset_path = config['GLOBAL']['dataset_path']

if len(sys.argv) == 3 and str(sys.argv[1]) == 'unet':

    if str(sys.argv[2]) == 'train':

        net = unet.TFNetwork(config)
        x_train, y_train, x_val, y_val = dataLoader.train_unet(dataset_path)
        net.train(x_train, y_train, x_val, y_val, display=True)

if len(sys.argv) == 3 and str(sys.argv[1]) == 'sr':

    if str(sys.argv[2]) == 'train':
        print(0)
        # x_train_noised, y_train, x_val_noised, y_val = dataLoader.train_sr(dataset_path)
        #
        # # Extract Feature Map from UNet
        # unet_model_path = '/home/xiaojianxu/gan/experiment/dec31/unet_ver3/final/model.cpkt'
        # net = unet.TFNetwork()
        # print('Extract Feature of Training data')
        # x_train_feature = net.predict(unet_model_path, x_train_noised, 1, True, batchsize=48)
        # print('Extract Feature of Validation data')
        # x_val_feature = net.predict(unet_model_path, x_val_noised, 1, True, batchsize=1)
        #
        # # Concatenate all Feature
        # x_train = np.concatenate([x_train_noised, x_train_feature], -1)
        # x_val = np.concatenate([x_val_noised, x_val_feature], -1)
        #
        # # Begin Training
        # net = sr.KerasNetwork(config, channel=2, num_gpu=num_gpu)
        # net.train(x_train, y_train, x_val, y_val)
