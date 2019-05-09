from datasets.mri_data_3d import MRIData3D
from datasets.mri_data_4d import MRIData4D
from datasets.jiaming_may3 import JMMay3

from methods.unet import UNet3D
from methods.unet_lstm import UNet3DLSTM

import os
import configparser
import logging

config = configparser.ConfigParser()
config.read('config.ini')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
os.environ["CUDA_VISIBLE_DEVICES"] = config['Setting']['gpu_index']
os.environ["OMP_NUM_THREADS"] = "4"

#####################
# Load Dataset
#####################
dataset_dict = {
    'MRIData3D': MRIData3D,
    'MRIData4D': MRIData4D,
    'JMMay3': JMMay3,
}

dataset = dataset_dict[config['Dataset']['dataset']](config)

#####################
# Load Model
#####################
model_dict = {
    'UNet3D': UNet3D,
    'UNet3DLSTM': UNet3DLSTM
}

model = model_dict[config['Setting']['model']](config)

#####################
# Main Processing
#####################
phase = config['Setting']['phase']

if phase == 'train':
    model.train(dataset.train, dataset.valid)

if phase == 'test':
    model.test(dataset.test)
