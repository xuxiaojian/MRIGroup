import tensorflow as tf
import config
from net import unet,sr

network_dict = {
    'unet': unet.KerasNetwork,
    'sr': sr.KerasNetwork,
    }

network_dict[tf.flags.FLAGS.network]()(tf.flags.FLAGS.mode)
