import tensorflow as tf
import config
from net import unet

network_dict = {'unet': unet.KerasNetwork}
network_dict[tf.flags.FLAGS.network]()(tf.flags.FLAGS.mode)
