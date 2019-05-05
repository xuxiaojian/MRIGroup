from .base import DatasetBase
import configparser
import tensorflow as tf
import h5py
import logging
import glob
import numpy as np


class MRIData3D:
    def __init__(self, config: configparser.ConfigParser):
        scan_lines = config['Dataset']['scan_lines']

        train_index = np.fromstring(config['Dataset']['train_index'], dtype=np.int, sep=',').tolist()
        valid_index = np.fromstring(config['Dataset']['valid_index'], dtype=np.int, sep=',').tolist()
        test_index = np.fromstring(config['Dataset']['test_index'], dtype=np.int, sep=',').tolist()
        sample_index = np.fromstring(config['Dataset']['sample_index'], dtype=np.int, sep=',').tolist()

        logging.root.info('Train Data:')
        self.train = MRIData3DBase(file_index=train_index, scan_lines=scan_lines, sample_index=sample_index, is_shuffle=True)
        logging.root.info('Valid Data:')
        self.valid = MRIData3DBase(file_index=valid_index, scan_lines=scan_lines, sample_index=sample_index, is_shuffle=False)
        logging.root.info('test Data:')
        self.test = MRIData3DBase(file_index=test_index, scan_lines=scan_lines, sample_index=sample_index, is_shuffle=False)


class MRIData3DBase(DatasetBase):
    def __init__(self, file_index: list,
                 scan_lines: str = '400',
                 sample_index: list = (16, 21, 27, 30),
                 is_shuffle: bool = True,
                 root_path='/export/project/gan.weijie/dataset/mri_source/'
                 ):

        logging.root.info("[MRIData Object]")

        # Read Source H5 File Pointer
        self.x_file = []; self.y_file = []
        self.sample_index = sample_index

        x_file_path = glob.glob(root_path + '*/MCNUFFT_' + scan_lines + '*.h5'); x_file_path.sort()
        y_file_path = glob.glob(root_path + '*/CS_2000' + '*.h5'); y_file_path.sort()

        for i in file_index:
            logging.root.info("Loading H5File Pointer in Path: " + x_file_path[i] + '.')
            self.x_file.append(h5py.File(x_file_path[i], 'r'))
            self.y_file.append(h5py.File(y_file_path[i], 'r'))

        _, self.phase, self.height, self.width = self.x_file[0]['recon_MCNUFFT'].shape
        self.channel = 1

        # Construct dataset
        output_shape = (self.phase, self.height, self.width, self.channel)
        super().__init__(is_shuffle=is_shuffle, indexes_length=2, x_shape=output_shape, y_shape=output_shape)

    def get_dataset_len(self):
        output_len = 0
        for i in self.x_file:
            output_len += i['recon_MCNUFFT'].shape[0]
        return output_len

    def get_sample_len(self):
        return len(self.sample_index)

    def dataset_generator(self):
        for file_index in range(len(self.x_file)):
            file_batches = self.x_file[file_index]['recon_MCNUFFT'].shape[0]

            for batch_index in range(file_batches):
                yield (file_index, batch_index)

    def sample_generator(self):
        for batch_index in self.sample_index:
            yield (0, batch_index)

    def read_data(self, indexes):
        file_index, batch_index = indexes

        def transform(input_np):
            output_np = np.swapaxes(input_np, 1, 2)

            for i in range(self.phase):
                output_np[i] -= np.amin(output_np[i])
                output_np[i] /= np.amax(output_np[i])
            # output_np -= np.amin(output_np)
            # output_np /= np.amax(output_np)

            output_np = np.expand_dims(output_np, -1)
            return output_np

        return transform(self.x_file[file_index]['recon_MCNUFFT'][batch_index]), \
            transform(self.y_file[file_index]['recon_CS'][batch_index])
