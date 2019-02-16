import tensorflow as tf


# ###############################
#       USER PATH
# ###############################
user_path = '/export/project/gan.weijie/'
# ###############################
# ###############################

# ###############################
#       GLOBAL
# ###############################
config_name = 'global_'

tf.flags.DEFINE_string(config_name + 'gpu_index', '3', 'Choice what GPUs tf can use')
tf.flags.DEFINE_string(config_name + 'network', 'sr', '')
tf.flags.DEFINE_string(config_name + 'mode', 'train', '')
tf.flags.DEFINE_string(config_name + 'output_path', user_path + '/experiment/feb14/sr_ver1/', '')

tf.flags.DEFINE_bool(config_name + 'debug', False, '')

tf.flags.DEFINE_string(config_name + 'root_path', user_path + 'data/', '')
tf.flags.DEFINE_string(config_name + 'dataset_type', 'mri_healthy_liver', '')

tf.flags.DEFINE_list(config_name + 'index_valid_mat', [9], 'valid when data_type is 0')
tf.flags.DEFINE_list(config_name + 'index_valid_images', [15, 30], 'valid when data_type is 0')

# ###############################
#       SCADEC
# ###############################
config_name = 'scadec_'

# DATA RELEVANT
tf.flags.DEFINE_list(config_name + 'index_train_mat', [1, 2, 3, 4, 5, 6, 7, 8], 'valid when data_type is 0')
tf.flags.DEFINE_list(config_name + 'index_train_images', [15], 'valid when data_type is 0')

# TRANING RELEVANT
tf.flags.DEFINE_integer(config_name + 'batch_size', 4, '')
tf.flags.DEFINE_integer(config_name + 'epoch', 30, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save', 100, '')
tf.flags.DEFINE_float(config_name + 'learning_rate', 0.001, '')

# NETWORK RELEVANT
tf.flags.DEFINE_integer(config_name + 'level', 4, '')
tf.flags.DEFINE_integer(config_name + 'root_filters', 32, '')
tf.flags.DEFINE_float(config_name + 'dropout', 0.1, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size', 3, '')
tf.flags.DEFINE_integer(config_name + 'conv_time', 2, '')

# ###############################
#       UNET
# ###############################
config_name = 'unet_'

# DATA RELEVANT
tf.flags.DEFINE_string(config_name + 'data_type', '0',  'KEEP STRING FORMAT: '
                                                        '0 means source mat file, '
                                                        '1 means old liver train mat file')

tf.flags.DEFINE_list(config_name + 'index_train_mat', [1, 2, 3, 4, 5, 6, 7, 8], 'valid when data_type is 0')
tf.flags.DEFINE_list(config_name + 'index_train_images', [15], 'valid when data_type is 0')

# TRANING RELEVANT
tf.flags.DEFINE_integer(config_name + 'batch_size', 4, '')
tf.flags.DEFINE_integer(config_name + 'epoch', 30, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save', 100, '')
tf.flags.DEFINE_float(config_name + 'learning_rate', 0.001, '')

# NETWORK RELEVANT
tf.flags.DEFINE_string(config_name + 'loss_type', '0', 'KEEP STRING FORMAT:  '
                                                       '0: mse,  '
                                                       '1: mse + old/wrong tv, '
                                                       '2: mse + new tv'
                                                       '3: mse + tv only for recon')

tf.flags.DEFINE_float(config_name + 'tv_lambda', 0.001, 'valid when loss contain regularization term')

tf.flags.DEFINE_list(config_name + 'input_shape', [None, None, 1], '')
tf.flags.DEFINE_integer(config_name + 'level', 4, '')
tf.flags.DEFINE_integer(config_name + 'root_filters', 32, '')
tf.flags.DEFINE_float(config_name + 'dropout', 0.1, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size', 3, '')
tf.flags.DEFINE_integer(config_name + 'conv_time', 2, '')

# ###############################
#       SUPER RESOLUTION
# ###############################
config_name = 'sr_'

# DATA RELEVANT
tf.flags.DEFINE_string(config_name + 'data_type', '3',  'KEEP STRING FORMAT: '
                                                        '0 means source mat file, '
                                                        '1 means old liver train mat file')

tf.flags.DEFINE_string(config_name + 'unet_model_path',
                       user_path + 'experiment/feb1/unet_ver4/' + 'final_weight.h5', '')

tf.flags.DEFINE_list(config_name + 'index_train_mat', [1, 2, 3, 4, 5, 6, 7, 8], 'valid when data_type is 0')
tf.flags.DEFINE_list(config_name + 'index_train_images', [20], 'valid when data_type is 0')

tf.flags.DEFINE_integer(config_name + 'crop_window_size', 60, '')
tf.flags.DEFINE_integer(config_name + 'crop_step', 20, '')

# TRANING RELEVANT
tf.flags.DEFINE_integer(config_name + 'batch_size', 16, '')
tf.flags.DEFINE_integer(config_name + 'epoch', 1000, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save', 100, '')
tf.flags.DEFINE_float(config_name + 'learning_rate', 0.0001, '')

# NETWORK RELEVANT
tf.flags.DEFINE_list(config_name + 'input_shape', [None, None, 2], '')
tf.flags.DEFINE_integer(config_name + 'filters', 128, '')
tf.flags.DEFINE_integer(config_name + 'num_rblocks', 32, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size_rblocks', 3, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size_io', 9, '')
# ###############################
# ###############################

# ###############################
#       Set up GPUs Environment
# ###############################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = tf.flags.FLAGS.global_gpu_index
import numpy as np
tf.flags.DEFINE_integer('global_gpu_num', np.fromstring(tf.flags.FLAGS.global_gpu_index,
                                                        dtype=np.int, sep=',').shape[0], '')
# ###############################
# ###############################
