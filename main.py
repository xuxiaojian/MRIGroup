import os
import configparser
from method.unet import Net2D
from data import loader
import numpy as np
from method.tfbase import config_to_markdown_table, TFTrainer

######################
# Load config file
######################
config = configparser.ConfigParser()
config.read('config.ini')

######################
# Set GPU index
######################
os.environ["CUDA_VISIBLE_DEVICES"] = config['global']['gpu_index']

config_info = config_to_markdown_table(config._sections['global'], 'global')
config_info = config_info + config_to_markdown_table(config._sections['data'], 'data')
config_info = config_info + config_to_markdown_table(config._sections['train'], 'train')

######################
# Load Train Data
######################
root_path = config['data']['root_path']
type_ = config['data']['type']

train_index = np.fromstring(config['data']['train_index'], dtype=np.int, sep=',').tolist()
valid_index = np.fromstring(config['data']['valid_index'], dtype=np.int, sep=',').tolist()
img_index = np.fromstring(config['data']['img_index'], dtype=np.int, sep=',').tolist()

is_patch = bool(config['data']['is_patch'])
patch_size = int(config['data']['patch_size'])
patch_step = int(config['data']['patch_step'])

train_x, train_y, train_x_imgs, train_y_imgs = loader.mri(root_path=root_path, type_=type_, data_index=train_index,
                                                          patch_size=patch_size, patch_step=patch_step,
                                                          is_patch=is_patch, img_index=img_index)

valid_x, valid_y, valid_x_imgs, valid_y_imgs = loader.mri(root_path=root_path, type_=type_, data_index=valid_index,
                                                          patch_size=patch_size, patch_step=patch_step,
                                                          is_patch=is_patch, img_index=img_index)

print("train_x: ", train_x.shape)
print("train_y: ", train_y.shape)
print("train_x_imgs: ", train_x_imgs.shape)
print("train_y_imgs: ", train_y_imgs.shape)
print("valid_x: ", valid_x.shape)
print("valid_y: ", valid_y.shape)
print("valid_x_imgs: ", valid_x_imgs.shape)
print("valid_y_imgs: ", valid_y_imgs.shape)

net = Net2D(config)

lr = float(config['train']['lr'])
dropout_rate = float(config['train']['dropout_rate'])

path = config['train']['path']
batch_size = int(config['train']['batch_size'])
train_epoch = int(config['train']['train_epoch'])
save_epoch = int(config['train']['save_epoch'])

config_info = config_info + config_to_markdown_table(config._sections['2d-unet'], '2d-unet')
trainer = TFTrainer(net, path, config_info, lr=lr, batch_size=batch_size, train_epoch=train_epoch,
                    save_epoch=save_epoch, dropout_value=dropout_rate)
trainer.run(train_x, train_y, valid_x, valid_y, train_x_imgs, train_y_imgs, valid_x_imgs, valid_y_imgs)
