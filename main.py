import os
import configparser
from method.unet import Net2D, Net3D
from data.tools import set_logging, new_folder
from data.loader import source_3d
import numpy as np
from method.tfbase import config_to_markdown_table, TFTrainer
import scipy.io as sio
import logging

config = configparser.ConfigParser()  # Load config file
config.read('config.ini')

config_info = config_to_markdown_table(config._sections['global'], 'global')
config_info = config_info + config_to_markdown_table(config._sections['data'], 'data')
config_info = config_info + config_to_markdown_table(config._sections['train'], 'train')

config_info = config_info + config_to_markdown_table(config._sections[config['global']['method']],
                                                     config['global']['method'])

######################
# Set Global Parameter
######################
gpu_index = config['global']['gpu_index']
method = config['global']['method']
phase = config['global']['phase']

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index  # Set GPU index

save_path = config['train']['save_path']
model_path = config['test']['model_path']
if phase == 'train':
    new_folder(save_path)  # New output folder
    set_logging(save_path)  # Set Logging Module
else:
    set_logging(model_path)  # Set Logging Module

######################
# Load Data
######################
root_path = config['data']['root_path']
scanlines = config['data']['scanlines']

train_index = np.fromstring(config['data']['train_index'], dtype=np.int, sep=',').tolist()
valid_index = np.fromstring(config['data']['valid_index'], dtype=np.int, sep=',').tolist()
test_index = np.fromstring(config['data']['test_index'], dtype=np.int, sep=',').tolist()

type_ = config['data']['type_']
is_patch = bool(int(config['data']['is_patch']))
patch_size = int(config['data']['patch_size'])
patch_step = int(config['data']['patch_step'])

load_data_dict = {
    'source_3d': source_3d,
}

if phase == 'train':
    logging.info("Loading train data. Number of file is [%d]." % train_index.__len__())
    train_x, train_y, train_x_imgs, train_y_imgs = load_data_dict[type_](root_path, train_index, scanlines, type_, is_patch, patch_size, patch_step)
    logging.info("Shape of train data: " + str(train_x.shape) + ". Shape of imgs of train data: " + str(train_x_imgs.shape))

    logging.info("Loading valid data. Number of file is [%d]." % valid_index.__len__())
    valid_x, valid_y, valid_x_imgs, valid_y_imgs = load_data_dict[type_](root_path, valid_index, scanlines, type_, is_patch, patch_size, patch_step)
    logging.info("Shape of valid data: " + str(valid_x.shape) + ". Shape of imgs of valid data: " + str(valid_x_imgs.shape))

if phase == 'test':
    logging.info("Loading test data. Number of file is [%d]." % test_index.__len__())
    test_x, test_y, _, _ = load_data_dict[type_](root_path, test_index, scanlines, type_, is_patch, patch_size, patch_step)
    logging.info("Shape of test data: " + str(test_x.shape))

######################
# Method
######################
net_dict = {
    "2d-unet": Net2D,
    "3d-unet": Net3D,
}

net = net_dict[method](config)

if phase == 'train':
    lr = float(config['train']['lr'])
    dropout_rate = float(config['train']['dropout_rate'])

    batch_size = int(config['train']['batch_size'])
    train_epoch = int(config['train']['train_epoch'])
    save_epoch = int(config['train']['save_epoch'])

    trainer = TFTrainer(net, save_path, config_info, lr=lr, batch_size=batch_size, train_epoch=train_epoch,
                        save_epoch=save_epoch, dropout_value=dropout_rate)
    trainer.run(train_x, train_y, valid_x, valid_y, train_x_imgs, train_y_imgs, valid_x_imgs, valid_y_imgs)

if phase == 'test':
    model_path = config['test']['model_path']
    batch_size = int(config['test']['batch_size'])

    predict = net.predict(test_x, batch_size, model_path)

    sio.savemat(model_path + 'test.mat', {
        "x": test_x,
        "y": test_y,
        "pre": predict,
    })
