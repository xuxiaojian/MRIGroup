import os
import configparser
from method.unet import Net2D, Net3D, ResNet3D, TorchNet3D
from method.sr import SRNet3D, SRNet2D
from data.tools import set_logging, new_folder
from data.loader import source_3d, source_sr_3d, liver_crop_3d, source_sr_2d, MRIData, liver_combine
import numpy as np
from method.tfbase import config_to_markdown_table, TFTrainer
import scipy.io as sio
import logging
import subprocess
import PIL
import shutil
from torch.utils.data import DataLoader
import torch

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

root_path = config['data']['root_path']
scanlines = config['data']['scanlines']

train_index = np.fromstring(config['data']['train_index'], dtype=np.int, sep=',').tolist()
valid_index = np.fromstring(config['data']['valid_index'], dtype=np.int, sep=',').tolist()
imgs_index = np.fromstring(config['data']['imgs_index'], dtype=np.int, sep=',').tolist()
test_index = np.fromstring(config['data']['test_index'], dtype=np.int, sep=',').tolist()

type_ = config['data']['type_']
is_patch = bool(int(config['data']['is_patch']))
patch_size = int(config['data']['patch_size'])
patch_step = int(config['data']['patch_step'])

if "torch" not in method:
    ######################
    # Load Data
    ######################
    load_data_dict = {
        'source_3d': source_3d,
        'source_sr_3d': source_sr_3d,
        'liver_crop_3d': liver_crop_3d,
        'source_sr_2d': source_sr_2d,
        'liver_combine': liver_combine,
    }

    if phase == 'train':
        logging.info("Loading train data. Number of file is [%d]." % train_index.__len__())
        train_x, train_y, train_x_imgs, train_y_imgs = load_data_dict[type_](root_path, train_index, imgs_index, scanlines, is_patch, patch_size, patch_step)
        logging.info("Shape of train_x data: " + str(train_x.shape) + ". Shape of imgs of train_x data: " + str(train_x_imgs.shape))
        logging.info("Shape of train_y data: " + str(train_y.shape) + ". Shape of imgs of train_y data: " + str(train_y_imgs.shape))

        logging.info("Loading valid data. Number of file is [%d]." % valid_index.__len__())
        valid_x, valid_y, valid_x_imgs, valid_y_imgs = load_data_dict[type_](root_path, valid_index, imgs_index, scanlines, is_patch, patch_size, patch_step)
        logging.info("Shape of valid data: " + str(valid_x.shape) + ". Shape of imgs of valid data: " + str(valid_x_imgs.shape))

    if phase == 'test':
        logging.info("Loading test data. Number of file is [%d]." % test_index.__len__())
        test_x, test_y, _, _ = load_data_dict[type_](root_path, test_index, imgs_index, scanlines, is_patch, patch_size, patch_step)
        logging.info("Shape of test data: " + str(test_x.shape))

    ######################
    # Method
    ######################
    net_dict = {
        "2d-unet": Net2D,
        "3d-unet": Net3D,
        '3d-resunet': ResNet3D,
        '3d-sr': SRNet3D,
        '2d-sr': SRNet2D,
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

        predict = net.predict(test_x, test_y, batch_size, model_path)

        def save_tiff(imgs, path):
            if imgs.shape.__len__() == 5:
                batches, depths, width, height, channel = imgs.shape

                imgs_list = []
                for depth in range(depths):
                    for batch in range(batches):
                        imgs_list.append(PIL.Image.fromarray(np.squeeze(imgs[batch, depth, :, :, :])))

                imgs_list[0].save(path, save_all=True, append_images=imgs_list[1:])

            else:
                if imgs.shape.__len__() == 4:
                    batches, width, height, channel = imgs.shape

                    imgs_list = []
                    for batch in range(batches):
                        imgs_list.append(PIL.Image.fromarray(np.squeeze(imgs[batch, :, :, :])))

                    imgs_list[0].save(path, save_all=True, append_images=imgs_list[1:])

                else:
                    logging.root.error("Incorrect Output Dimension when .tiff file outputing")


        save_tiff(predict, model_path + 'predict.tiff')

        sio.savemat(model_path + 'predict.mat', {
            "pre": predict,
        })

else:
    # train_dataset = MRIData(mat_index=train_index, root_path=root_path, scanlines=scanlines)
    # # valid_dataser = MRIData(mat_index=valid_index, root_path=root_path, scanlines=scanlines)
    #
    # x, y = train_dataset[0]
    # x = torch.rand(1, 1, 10, 320, 320).cuda()
    # net = TorchNet3D(config)
    # net.cuda()
    # net.train()
    # print(net(x).shape)
    pass
