from datasets.mri_data_3d import MRIData3D
from datasets.jiaming_may3 import JMMay3
from methods.utilities import copytree_code, set_logging, new_folder, dict_to_markdown_table, save_predict
from methods.unet import unet_3d, unet_3d_lstm
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
os.environ["OMP_NUM_THREADS"] = "12"

if phase == 'train':
    copytree_code(src_path=code_folder, dst_path=experiment_folder + train_folder + '/')
    set_logging(target_path=experiment_folder + train_folder + '/')
    new_folder(target_path=experiment_folder + train_folder + '/model/')

test_save_folder = model_file + ' + ' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
if phase == 'test':
    new_folder(target_path=experiment_folder + test_folder + '/model/' + test_save_folder + '/')
    set_logging(target_path=experiment_folder + test_folder + '/model/' + test_save_folder + '/')

logging.root.info("Config Info: \n" + config_info)

#####################
# Load Dataset
#####################
dataset_dict = {
    'MRIData3D': MRIData3D,
    'JMMay3': JMMay3,
}

dataset = dataset_dict[config['Dataset']['dataset']](config)
#####################
# Main Processing
#####################
if phase == 'train':
    method = config['Train']['method']
    method_dict = {
        'UNet3D': unet_3d,
        'UNet3DLSTM': unet_3d_lstm
                   }

    train_batch_size = int(config['Train']['batch_size'])
    learning_rate = float(config['Train']['learning_rate'])
    dropout_rate = float(config['Train']['dropout_rate'])

    train_epoch = int(config['Train']['train_epoch'])
    model_save_epoch = int(config['Train']['model_save_epoch'])

    model = method_dict[method](config, method)
    model.summary()
    tf.keras.utils.plot_model(model, to_file=experiment_folder + train_folder + '/model.png', show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=[psnr_tf, ssim_tf])

    model.fit(dataset.train.tf_dataset.batch(train_batch_size),
              epochs=train_epoch,
              steps_per_epoch=dataset.train.get_dataset_len() // train_batch_size,
              validation_data=dataset.valid.tf_dataset.batch(train_batch_size),
              validation_steps=dataset.valid.get_dataset_len() // train_batch_size,
              callbacks=[
              tf.keras.callbacks.ModelCheckpoint(filepath=experiment_folder + train_folder + '/model/{epoch:02d}-{val_loss:.2f}.h5',
                                                 verbose=1, period=model_save_epoch),
              tf.keras.callbacks.ModelCheckpoint(filepath=experiment_folder + train_folder + '/model/psnr.best.h5',
                                                 verbose=1, save_best_only=True, monitor='val_psnr_tf', mode='max'),
              tf.keras.callbacks.ModelCheckpoint(filepath=experiment_folder + train_folder + '/model/ssim.best.h5',
                                                 verbose=1, save_best_only=True, monitor='val_ssim_tf', mode='max'),
              KerasCallBack(output_path=experiment_folder + train_folder + '/',
                            train_dataset=dataset.train.tf_sample,
                            valid_dataset=dataset.valid.tf_sample,
                            config_info=config_info),
              ]
              )

    imgs_predict = model.predict(dataset.test.tf_dataset.batch(train_batch_size), steps=dataset.test.get_dataset_len() // train_batch_size, verbose=1)
    eval_predict = model.evaluate(dataset.test.tf_dataset.batch(train_batch_size), steps=dataset.test.get_dataset_len() // train_batch_size, verbose=1)
    new_folder(experiment_folder + train_folder + '/model/after-training/')

    save_predict(imgs=imgs_predict,
                 evaluation=eval_predict,
                 target_name=config['Train']['name_predict_file'],
                 target_path=experiment_folder + train_folder + '/model/after-training/', slim=True)
