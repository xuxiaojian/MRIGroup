import configparser
import os
import sys
from net import unet

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']

if len(sys.argv) == 2 and str(sys.argv[1]) == 'unet':

    net = unet.KerasNetwork(config)
    net.train()
