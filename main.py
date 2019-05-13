from datasets.mri_data_3d import MRIData3D
from datasets.mri_data_4d import MRIData4D
from datasets.jiaming_may3 import JMMay3
from datasets.mri_data_3d_slim import MRIData3DSlim

from methods.unet import UNet3D
from methods.unet_lstm import UNet3DLSTM
from methods.unet_gan import UNet3dGAN
from methods.unet_phase_slim import UNet3DPhaseSlim

import os
import configparser
import logging
import tensorflow as tf

config = configparser.ConfigParser()
config.read('config.ini')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
os.environ["CUDA_VISIBLE_DEVICES"] = config['Setting']['gpu_index']
os.environ["OMP_NUM_THREADS"] = "4"

if bool(int(config['Setting']['is_eager'])):
    tf.enable_eager_execution()

#####################
# Load Dataset
#####################
dataset_dict = {
    'mri_data_3d': MRIData3D,
    'mri_data_4d': MRIData4D,
    'jiaming_may3': JMMay3,
    'mri_data_3d_slim': MRIData3DSlim,
}

dataset = dataset_dict[config['Dataset']['dataset']](config)

#####################
# Load Model
#####################
model_dict = {
    'unet': UNet3D,
    'unet_lstm': UNet3DLSTM,
    'unet_gan': UNet3dGAN,
    'unet_phase_slim': UNet3DPhaseSlim,
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
