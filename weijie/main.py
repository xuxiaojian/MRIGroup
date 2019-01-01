import configparser
import os
import sys
from net import unet
from util import dataLoader

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']
dataset_path = config['GLOBAL']['dataset_path']

if len(sys.argv) == 3 and str(sys.argv[1]) == 'unet':

    if str(sys.argv[2]) == 'train':

        x_train, y_train, x_val, y_val = dataLoader.train_unet(dataset_path)
        print(x_train.shape)
        net = unet.KerasNetwork(config, (320, 320, 1))
        net.train(x_train, y_train, x_val, y_val)
