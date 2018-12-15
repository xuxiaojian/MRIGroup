import configparser
import os
import sys
from net import edsr, edsr_mix

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']

sr_nets = {
    'edsr': edsr,
    'edsr_mix': edsr_mix,
}

if len(sys.argv) == 2 and str(sys.argv[1]) == 'sr':

    net = sr_nets[config['SR']['method']].KerasNetwork(config)
    net.train()
