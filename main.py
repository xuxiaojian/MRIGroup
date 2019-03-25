import os
import configparser
from method.unet import Net2D, Net3D
from data.loader import original, original_structural_patch_3d, liver_crop, liver_crop_structural_patch_3d
import numpy as np
from method.tfbase import config_to_markdown_table, TFTrainer
import scipy.io as sio

######################
# Load config file
######################
config = configparser.ConfigParser()
config.read('config.ini')

phase = config['global']['phase']

######################
# Set GPU index
######################
os.environ["CUDA_VISIBLE_DEVICES"] = config['global']['gpu_index']

######################
# Set config_info
######################

config_info = config_to_markdown_table(config._sections['global'], 'global')
config_info = config_info + config_to_markdown_table(config._sections['data'], 'data')
config_info = config_info + config_to_markdown_table(config._sections['train'], 'train')

config_info = config_info + config_to_markdown_table(config._sections[config['global']['method']],
                                                     config['global']['method'])
######################
# Load Train Data
######################
root_path = config['data']['root_path']

train_index = np.fromstring(config['data']['train_index'], dtype=np.int, sep=',').tolist()
valid_index = np.fromstring(config['data']['valid_index'], dtype=np.int, sep=',').tolist()
test_index = np.fromstring(config['data']['test_index'], dtype=np.int, sep=',').tolist()
img_index = np.fromstring(config['data']['img_index'], dtype=np.int, sep=',').tolist()

is_patch = bool(int(config['data']['is_patch']))
patch_size = int(config['data']['patch_size'])
patch_step = int(config['data']['patch_step'])

loader_dict = {
    "original": original,
    "liver_crop": liver_crop,
    "original_structural_patch_3d": original_structural_patch_3d,
    "liver_crop_structural_patch_3d": liver_crop_structural_patch_3d,
}

if phase == 'train':
    train_x, train_y, train_x_imgs, train_y_imgs = loader_dict[config['data']['type']](
        root_path=root_path, data_index=train_index, patch_size=patch_size, patch_step=patch_step,
        is_patch=is_patch, img_index=img_index)

    valid_x, valid_y, valid_x_imgs, valid_y_imgs = loader_dict[config['data']['type']](
        root_path=root_path, data_index=valid_index, patch_size=patch_size, patch_step=patch_step,
        is_patch=is_patch, img_index=img_index)

    print("Training Data Shape: ", train_x.shape, "Training Label Shape: ", train_y.shape)
    print("Validation Data Shape: ", valid_x.shape, "Validation Label Shape: ", valid_y.shape)

if phase == 'test':
    test_x, test_y, _, _ = loader_dict[config['data']['type']](
        root_path=root_path, data_index=test_index, patch_size=patch_size, patch_step=patch_step,
        is_patch=is_patch, img_index=img_index)

    print("Testing Data Shape: ", test_x.shape, "Testing Label Shape: ", test_y.shape)

######################
# Build Network
######################
net_dict = {
    "2d-unet": Net2D,
    "3d-unet": Net3D,
}

net = net_dict[config['global']['method']](config)

if phase == 'train':
    ######################
    # Start Training
    ######################

    lr = float(config['train']['lr'])
    dropout_rate = float(config['train']['dropout_rate'])

    path = config['train']['path']
    batch_size = int(config['train']['batch_size'])
    train_epoch = int(config['train']['train_epoch'])
    save_epoch = int(config['train']['save_epoch'])

    trainer = TFTrainer(net, path, config_info, lr=lr, batch_size=batch_size, train_epoch=train_epoch,
                        save_epoch=save_epoch, dropout_value=dropout_rate)
    trainer.run(train_x, train_y, valid_x, valid_y, train_x_imgs, train_y_imgs, valid_x_imgs, valid_y_imgs)

if phase == 'test':
    ######################
    # Start Test
    ######################
    model_path = config['test']['model_path']
    batch_size = int(config['test']['batch_size'])

    predict = net.predict(test_x, batch_size, model_path)

    sio.savemat(model_path + 'test.mat', {
        "x": test_x,
        "y": test_y,
        "pre": predict,
    })
