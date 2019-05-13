from .unet import UNet3D
from datasets.base import DatasetBase
import tensorflow as tf
import numpy as np


class UNet3DPhaseSlim(UNet3D):
    def __init__(self, config):
        super().__init__(config=config)

    def save_test(self, metrics, outputs, dataset: DatasetBase, save_path):
        output_file = self.config['Test']['output_file']

        def transform(input_):
            output_ = input_[:, 1]
            output_ = np.squeeze(output_)

            width = output_.shape[-1]
            height = output_.shape[-2]
            output_.shape = [-1, height, width]

            return output_

        predict = np.concatenate(outputs, 0)
        with tf.Session() as sess:
            x, y = sess.run(dataset.tf_dataset.batch(dataset.dataset_len()).make_one_shot_iterator().get_next())

        x = transform(x)
        y = transform(y)
        predict = transform(predict)

        write_path = save_path + output_file

        self.write_test(x, y, predict, write_path)
