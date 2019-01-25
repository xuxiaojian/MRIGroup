import tensorflow as tf


# ###############################
#       USER PATH
# ###############################
user_path = '/export/project/gan.weijie/'
# ###############################
# ###############################

# ###############################
#       USER DEFINE PAREMETERS
# ###############################
tf.flags.DEFINE_string('gpu_index', '1,2,3', 'Choice what GPUs tf can use')
tf.flags.DEFINE_string('network', 'sr', '')
tf.flags.DEFINE_string('mode', 'train', '')
tf.flags.DEFINE_string('output_path', user_path + '/experiment/jan25/temp/', '')

tf.flags.DEFINE_bool('debug', False, '')
# ###############################
# ###############################

tf.flags.DEFINE_string('root_path', user_path + 'data/', '')
tf.flags.DEFINE_string('dataset_type', 'mri_healthy_liver', '')

tf.flags.DEFINE_list('index_valid_mat', [9], '')
tf.flags.DEFINE_list('index_valid_images', [15, 30, 35], '')

# ###############################
#       UNET
# ###############################
config_name = 'unet_'

# DATA RELEVANT
tf.flags.DEFINE_list(config_name + 'index_train_mat', [1, 2, 3, 4], '')
tf.flags.DEFINE_list(config_name + 'index_train_images', [15], '')

# TRANING RELEVANT
tf.flags.DEFINE_integer(config_name + 'batch_size', 2, '')
tf.flags.DEFINE_integer(config_name + 'epoch', 300, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save_model', 50, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save_val', 10, '')
tf.flags.DEFINE_float(config_name + 'learning_rate', 0.001, '')

# NETWORK RELEVANT
tf.flags.DEFINE_list(config_name + 'input_shape', [None, None, 1], '')
tf.flags.DEFINE_integer(config_name + 'level', 5, '')
tf.flags.DEFINE_integer(config_name + 'root_filters', 64, '')
tf.flags.DEFINE_float(config_name + 'dropout', 0.1, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size', 3, '')
tf.flags.DEFINE_integer(config_name + 'conv_time', 2, '')
# ###############################
# ###############################

# ###############################
#       SUPER RESOLUTION
# ###############################
config_name = 'sr_'

# DATA RELEVANT
tf.flags.DEFINE_string(config_name + 'unet_model_path',
                       user_path + '/experiment/jan24/unet_1to4liver_droup0.9_large_net/' + 'model/final_weight.h5', '')

tf.flags.DEFINE_list(config_name + 'index_train_mat', [5, 6, 7, 8], '')
tf.flags.DEFINE_list(config_name + 'index_train_images', [15], '')

tf.flags.DEFINE_integer(config_name + 'crop_window_size', 60, '')
tf.flags.DEFINE_integer(config_name + 'crop_step', 20, '')

# TRANING RELEVANT
tf.flags.DEFINE_integer(config_name + 'batch_size', 128, '')
tf.flags.DEFINE_integer(config_name + 'epoch', 300, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save_model', 50, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save_val', 10, '')
tf.flags.DEFINE_float(config_name + 'learning_rate', 0.0001, '')

# NETWORK RELEVANT
tf.flags.DEFINE_list(config_name + 'input_shape', [None, None, 2], '')
tf.flags.DEFINE_integer(config_name + 'filters', 64, '')
tf.flags.DEFINE_integer(config_name + 'num_rblocks', 16, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size_rblocks', 3, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size_io', 9, '')
# ###############################
# ###############################

# ###############################
#       Set up GPUs Environment
# ###############################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = tf.flags.FLAGS.gpu_index
import numpy as np
tf.flags.DEFINE_integer('gpu_num', np.fromstring(tf.flags.FLAGS.gpu_index, dtype=np.int, sep=',').shape[0], '')
# ###############################
# ###############################