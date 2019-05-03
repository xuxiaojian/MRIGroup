from dataset import MRIData
from methods.utilities import copytree_code, set_logging, new_folder, dict_to_markdown_table, save_predict
from methods.unet import unet_3d
import os
import logging
import configparser
import numpy as np
import datetime
from methods.customed_keras import ssim_tf, psnr_tf, KerasCallBack
import tensorflow as tf

#####################
# Environment Setting
#####################
config = configparser.ConfigParser()
config.read('config.ini')

config_info = dict_to_markdown_table(config._sections['Setting'], 'Setting')
config_info = config_info + dict_to_markdown_table(config._sections['Dataset'], 'Dataset')
config_info = config_info + dict_to_markdown_table(config._sections['Train'], 'Train')
config_info = config_info + dict_to_markdown_table(config._sections['Test'], 'Test')
config_info = config_info + dict_to_markdown_table(config._sections[config['Train']['method']], config['Train']['method'])

gpu_index = config['Setting']['gpu_index']
code_folder = config['Setting']['code_folder']

experiment_folder = config['Setting']['experiment_folder']
train_folder = config['Setting']['train_folder']
test_folder = config['Setting']['test_folder']

model_file = config['Test']['model_file']

phase = config['Setting']['phase']

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

if phase == 'train':
    copytree_code(src_path=code_folder, dst_path=experiment_folder + train_folder + '/')
    set_logging(target_path=experiment_folder + train_folder + '/')
    new_folder(target_path=experiment_folder + train_folder + '/model/')

test_save_folder = model_file + ' + ' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
if phase == 'test':
    new_folder(target_path=experiment_folder + test_folder + '/model/' + test_save_folder + '/')
    set_logging(target_path=experiment_folder + test_folder + '/model/' + test_save_folder + '/')

logging.root.info("Config Info: ")
logging.root.info(config_info)

#####################
# Load Dataset
#####################
root_path = config['Dataset']['root_path']
scan_lines = config['Dataset']['scan_lines']
is_temporal = bool(int(config['Dataset']['is_temporal']))
time_step = int(config['Dataset']['time_step'])

train_index = np.fromstring(config['Dataset']['train_index'], dtype=np.int, sep=',').tolist()
valid_index = np.fromstring(config['Dataset']['valid_index'], dtype=np.int, sep=',').tolist()
test_index = np.fromstring(config['Dataset']['test_index'], dtype=np.int, sep=',').tolist()
sample_index = np.fromstring(config['Dataset']['sample_index'], dtype=np.int, sep=',').tolist()

logging.root.info('[Load Train Data]')
train_data = MRIData(train_index, scan_lines, root_path, sample_index, is_temporal=is_temporal, time_step=time_step)
logging.root.info('[Load Valid Data]')
valid_data = MRIData(valid_index, scan_lines, root_path, sample_index, is_temporal=is_temporal, time_step=time_step)
logging.root.info('[Load Test Data]')
test_data = MRIData(test_index, scan_lines, root_path, sample_index, is_temporal=is_temporal, time_step=time_step, is_shuffle=False)

#####################
# Main Processing
#####################
test_batch_size = int(config['Test']['batch_size'])

if phase == 'train':
    method = config['Train']['method']
    method_dict = {'UNet3D': unet_3d}

    train_batch_size = int(config['Train']['batch_size'])
    learning_rate = float(config['Train']['learning_rate'])
    dropout_rate = float(config['Train']['dropout_rate'])

    train_epoch = int(config['Train']['train_epoch'])
    model_save_epoch = int(config['Train']['model_save_epoch'])

    model = method_dict[method](config, method)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=[psnr_tf, ssim_tf])

    model.fit(train_data.tf_dataset.batch(train_batch_size),
              epochs=train_epoch,
              steps_per_epoch=train_data.dataset_len() // train_batch_size,
              validation_data=test_data.tf_dataset.batch(train_batch_size),
              validation_steps=test_data.dataset_len() // train_batch_size,
              callbacks=[KerasCallBack(output_path=experiment_folder + train_folder + '/',
                                       train_dataset=train_data.tf_sample,
                                       valid_dataset=valid_data.tf_sample,
                                       config_info=config_info),
                         tf.keras.callbacks.ModelCheckpoint(filepath=experiment_folder + train_folder + '/model/{epoch:02d}-{val_loss:.2f}.h5',
                                                            verbose=1,
                                                            period=model_save_epoch),
                         tf.keras.callbacks.ModelCheckpoint(filepath=experiment_folder + train_folder + '/model/psnr.best.h5',
                                                            verbose=1, save_best_only=True, monitor='val_psnr_tf', mode='max'),
                         tf.keras.callbacks.ModelCheckpoint(filepath=experiment_folder + train_folder + '/model/ssim.best.h5',
                                                            verbose=1, save_best_only=True, monitor='val_ssim_tf', mode='max')]
              )

    imgs_predict = model.predict(test_data.tf_dataset.batch(test_batch_size), steps=test_data.dataset_len() // test_batch_size, verbose=1)
    eval_predict = model.evaluate(test_data.tf_dataset.batch(test_batch_size), steps=test_data.dataset_len() // test_batch_size, verbose=1)
    new_folder(experiment_folder + train_folder + '/model/after-training/')

    save_predict(imgs=imgs_predict,
                 evaluation=eval_predict,
                 target_name='test_index:' + str(test_index),
                 target_path=experiment_folder + train_folder + '/model/after-training/', slim=True)
