import configparser
import os
import sys
import numpy as np

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']
dataset_path = config['GLOBAL']['dataset_path']

if len(sys.argv) == 2 and str(sys.argv[1]) == 'sr':

    from net import edsr
    net = edsr.KerasNetwork(config, dataset_path)
    net.train()
