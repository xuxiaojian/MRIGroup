import tensorflow as tf
import config
# from net import unet,sr
from net import sr

network_dict = {
    # 'unet': unet.KerasNetwork,
    'sr': sr.KerasNetwork,
    }

network_dict[tf.flags.FLAGS.global_network]()(tf.flags.FLAGS.global_mode)

# config = configparser.ConfigParser()
# config.read('config.ini')
#
# os.environ['CUDA_VISIBLE_DEVICES'] = config['global']['gpu_index']
#
# dearti.ScadecNet(config['global']['save_path'], config).train()
