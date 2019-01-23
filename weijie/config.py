import tensorflow as tf
import os

# GLOBAL
tf.flags.DEFINE_string('gpu_index', '2', 'Choice what GPUs tf can use')
tf.flags.DEFINE_string('network', 'unet', '')
tf.flags.DEFINE_string('mode', 'train', '')

# DATASET
tf.flags.DEFINE_string('root_path', '/home/xiaojianxu/gan/data/', '')
tf.flags.DEFINE_string('dataset_type', 'liver', '')

tf.flags.DEFINE_list('index_train_unet_mat', [1, 2, 3, 4], '')
tf.flags.DEFINE_list('index_train_unet_images', [15], '')

tf.flags.DEFINE_list('index_train_sr_mat', [5, 6], '')
tf.flags.DEFINE_list('index_train_sr_images', [15], '')

tf.flags.DEFINE_list('index_valid_mat', [9], '')
tf.flags.DEFINE_list('index_valid_images', [0, 1, 2, 3], '')

# UNET
tf.flags.DEFINE_string('unet_output_path', '/home/xiaojianxu/gan/experiment/jan22/unet_ver1/', '')
tf.flags.DEFINE_integer('unet_batch_size', 4, '')
tf.flags.DEFINE_integer('unet_epoch', 300, '')
tf.flags.DEFINE_integer('unet_epoch_save_model', 50, '')
tf.flags.DEFINE_float('unet_learning_rate', 0.001, '')

tf.flags.DEFINE_list('unet_net_input_shape', [None, None, 1], '')
tf.flags.DEFINE_integer('unet_net_level', 3, '')
tf.flags.DEFINE_integer('unet_net_root_filters', 32, '')
tf.flags.DEFINE_float('unet_net_dropout', 0.1, '')
tf.flags.DEFINE_integer('unet_net_kernel_size', 3, '')

# ############################
#       Set up GPUs
# ############################
os.environ['CUDA_VISIBLE_DEVICES'] = tf.flags.FLAGS.gpu_index
