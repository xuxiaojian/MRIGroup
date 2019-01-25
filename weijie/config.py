import tensorflow as tf

# ###############################
#       USER DEFINE PAREMETERS
# ###############################
user_path = '/export/project/gan.weijie/'

# GLOBAL
tf.flags.DEFINE_string('gpu_index', '3', 'Choice what GPUs tf can use')
tf.flags.DEFINE_string('network', 'unet', '')
tf.flags.DEFINE_string('mode', 'train', '')

tf.flags.DEFINE_string('root_path', user_path + 'data/', '')
tf.flags.DEFINE_string('dataset_type', 'mri_healthy_liver', '')

tf.flags.DEFINE_list('index_valid_mat', [9], '')
tf.flags.DEFINE_list('index_valid_images', [15, 30, 35], '')

# UNET
config_name = 'unet_'

tf.flags.DEFINE_list(config_name + 'index_train_mat', [1], '')
tf.flags.DEFINE_list(config_name + 'index_train_images', [15], '')

tf.flags.DEFINE_string(config_name + 'output_path',
                       user_path + '/experiment/jan24/temp/', '')
tf.flags.DEFINE_integer(config_name + 'batch_size', 2, '')
tf.flags.DEFINE_integer(config_name + 'epoch', 300, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save_model', 50, '')
tf.flags.DEFINE_integer(config_name + 'epoch_save_val', 10, '')
tf.flags.DEFINE_float(config_name + 'learning_rate', 0.001, '')

tf.flags.DEFINE_list(config_name + 'input_shape', [None, None, 1], '')
tf.flags.DEFINE_integer(config_name + 'level', 5, '')
tf.flags.DEFINE_integer(config_name + 'root_filters', 64, '')
tf.flags.DEFINE_float(config_name + 'dropout', 0.1, '')
tf.flags.DEFINE_integer(config_name + 'kernel_size', 3, '')
tf.flags.DEFINE_integer(config_name + 'conv_time', 2, '')

# SR
config_name = 'sr_'

tf.flags.DEFINE_list(config_name + 'index_train_mat', [4, 5, 6, 7], '')
tf.flags.DEFINE_list(config_name + 'index_train_images', [15], '')


# ###############################
#       Set up GPUs Environment
# ###############################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = tf.flags.FLAGS.gpu_index
import numpy as np
tf.flags.DEFINE_integer('gpu_num', np.fromstring(tf.flags.FLAGS.gpu_index, dtype=np.int, sep=',').shape[0], '')
