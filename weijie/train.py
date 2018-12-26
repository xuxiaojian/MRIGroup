import configparser
import os
import sys
from net import edsr, edsr_mix, srcnn, srcnn_mix, unet

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']

sr_nets = {
    'edsr': edsr,
    'edsr_mix': edsr_mix,
    'srcnn': srcnn,
    'srcnn_mix': srcnn_mix,
}

if len(sys.argv) == 2 and str(sys.argv[1]) == 'sr':

    net = sr_nets[config['SR']['method']].KerasNetwork(config)
    net.train()

if len(sys.argv) == 2 and str(sys.argv[1]) == 'unet':

    net = unet.KerasNetwork(config)
    net.train()
