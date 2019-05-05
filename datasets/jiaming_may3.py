from .base import DatasetBase
import configparser
import tensorflow as tf
import h5py
import numpy as np


class JMMay3:
    def __init__(self, config: configparser.ConfigParser):
        # noinspection SpellCheckingInspection
        src_h5file = h5py.File('/export/project/gan.weijie/dataset/Denoiser_Jiamings/GaussiantoClean_Training_Set_320_320_10_1_864_MRILiver_s1_scale0.5.mat', 'r')
        self.train = JMMay3Base(x_h5File=src_h5file['meas_tri'],
                                y_h5File=src_h5file['truth_tri'],
                                sample_index=[5, 10, 15, 20],
                                is_shuffle=True)

        self.valid = JMMay3Base(x_h5File=src_h5file['meas_val'],
                                y_h5File=src_h5file['truth_val'],
                                sample_index=[5, 10, 15, 20],
                                is_shuffle=False)

        self.test = JMMay3Base(x_h5File=src_h5file['meas_val'],
                               y_h5File=src_h5file['truth_val'],
                               sample_index=[5, 10, 15, 20],
                               is_shuffle=False)


class JMMay3Base(DatasetBase):
    def __init__(self, x_h5File, y_h5File, sample_index, is_shuffle):
        self.x_h5File = x_h5File
        self.y_h5File = y_h5File
        self.sample_index = sample_index

        _, self.channel, self.phase, self.width, self.height = self.x_h5File.shape
        output_shape = (self.phase, self.height, self.width, self.channel)
        super().__init__(is_shuffle=is_shuffle, indexes_length=[1], x_shape=output_shape, y_shape=output_shape)

    def get_dataset_len(self):
        return self.x_h5File.shape[0]

    def get_sample_len(self):
        return len(self.sample_index)

    def dataset_generator(self):
        dataset_batches = self.get_dataset_len()
        for batch_index in range(dataset_batches):
            yield (batch_index, )

    def sample_generator(self):
        for batch_index in self.sample_index:
            yield (batch_index, )

    def read_data(self, indexes):
        def transform(input_np):
            output_np = np.swapaxes(input_np, 0, 1)
            output_np = np.swapaxes(output_np, 1, 3)

            phase = output_np.shape[0]
            for i in range(phase):
                output_np[i] -= np.amin(output_np[i])
                output_np[i] /= np.amax(output_np[i])

            return output_np

        return transform(self.x_h5File[indexes]), transform(self.y_h5File[indexes])
