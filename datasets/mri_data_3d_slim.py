from .base import DatasetBase
import configparser
import h5py
import logging
import glob
import numpy as np


class MRIData3DSlim:
    def __init__(self, config: configparser.ConfigParser):
        scan_lines = config['Dataset']['scan_lines']

        train_index = np.fromstring(config['Dataset']['train_index'], dtype=np.int, sep=',').tolist()
        valid_index = np.fromstring(config['Dataset']['valid_index'], dtype=np.int, sep=',').tolist()
        test_index = np.fromstring(config['Dataset']['test_index'], dtype=np.int, sep=',').tolist()
        sample_index = np.fromstring(config['Dataset']['sample_index'], dtype=np.int, sep=',').tolist()
        is_liver_crop = bool(int(config['Dataset']['is_liver_crop']))

        logging.root.info('Train Data:')
        self.train = MRIData3DBaseSlim(file_index=train_index, scan_lines=scan_lines, sample_index=sample_index, is_shuffle=True, is_liver_crop=is_liver_crop)
        logging.root.info('Valid Data:')
        self.valid = MRIData3DBaseSlim(file_index=valid_index, scan_lines=scan_lines, sample_index=sample_index, is_shuffle=False, is_liver_crop=is_liver_crop)
        logging.root.info('test Data:')
        self.test = MRIData3DBaseSlim(file_index=test_index, scan_lines=scan_lines, sample_index=sample_index, is_shuffle=False, is_liver_crop=is_liver_crop)


class MRIData3DBaseSlim(DatasetBase):
    def __init__(self, file_index: list,
                 scan_lines: str = '400',
                 phase_step: int = 3,
                 sample_index: list = (16, 21, 27, 30),
                 is_shuffle: bool = True,
                 root_path='/export/project/gan.weijie/dataset/mri_source/',
                 is_liver_crop: bool = False,
                 ):

        logging.root.info("[MRIData Object]")

        # Read Source H5 File Pointer
        self.x_file = []; self.y_file = []
        self.file_index = file_index
        self.sample_index = sample_index
        self.phase_step = phase_step

        self.is_liver_crop = is_liver_crop
        self.liver_crop_mask = [
            [116, 244, 70, 198],
            [96, 224, 51, 179],
            [169, 297, 111, 239],
            [121, 249, 101, 229],
            [111, 239, 93, 221],
            [106, 234, 103, 231],
            [96, 224, 98, 226],
            [51, 179, 103, 231],
            [76, 204, 86, 214],
            [170, 298, 95, 223],
        ]

        x_file_path = glob.glob(root_path + '*/MCNUFFT_' + scan_lines + '*.h5'); x_file_path.sort()
        y_file_path = glob.glob(root_path + '*/CS_2000' + '*.h5'); y_file_path.sort()

        for i in file_index:
            logging.root.info("Loading H5File Pointer in Path: " + x_file_path[i] + '.')
            self.x_file.append(h5py.File(x_file_path[i], 'r'))
            self.y_file.append(h5py.File(y_file_path[i], 'r'))

        _, _, self.height, self.width = self.x_file[0]['recon_MCNUFFT'].shape
        self.channel = 1

        if is_liver_crop:
            self.height = 128; self.width = 128  # depends on the dimension of mask
        # Construct dataset
        output_shape = (self.phase_step, self.height, self.width, self.channel)
        super().__init__(is_shuffle=is_shuffle, indexes_length=2 + self.phase_step, x_shape=output_shape, y_shape=output_shape)

    def dataset_len(self):
        output_len = 0
        for i in self.x_file:
            output_len += (i['recon_MCNUFFT'].shape[0] * (i['recon_MCNUFFT'].shape[1] - 2 * int(self.phase_step / 2)))

        return output_len

    def sample_len(self):
        return len(self.sample_index)

    def dataset_generator(self):
        for file_index in range(len(self.x_file)):
            file_batches = self.x_file[file_index]['recon_MCNUFFT'].shape[0]
            file_phases = self.x_file[file_index]['recon_MCNUFFT'].shape[1]

            for batch_index in range(file_batches):
                for phase_index in range(int(self.phase_step / 2), file_phases - int(self.phase_step / 2), 1):
                    temporal_index = np.array([i for i in range(self.phase_step)]) + phase_index - int(self.phase_step / 2)
                    temporal_index = temporal_index.tolist()

                    yield [file_index, batch_index] + temporal_index

    def sample_generator(self):
        temporal_index = np.array([i for i in range(self.phase_step)])
        temporal_index = temporal_index.tolist()

        for batch_index in self.sample_index:
            yield [0, batch_index] + temporal_index

    def read_data(self, indexes):
        file_index = indexes[0]
        batch_index = indexes[1]
        phase_index = indexes[2:]

        def transform(input_np):
            output_np = np.swapaxes(input_np, 1, 2)
            if self.is_liver_crop:
                output_np = output_np[
                            :, self.liver_crop_mask[self.file_index[file_index]][0]:self.liver_crop_mask[self.file_index[file_index]][1],
                            self.liver_crop_mask[self.file_index[file_index]][2]:self.liver_crop_mask[self.file_index[file_index]][3]
                            ]

            for i in range(self.phase_step):
                output_np[i] -= np.amin(output_np[i])
                output_np[i] /= np.amax(output_np[i])

            output_np = np.expand_dims(output_np, -1)
            return output_np

        return transform(self.x_file[file_index]['recon_MCNUFFT'][batch_index, phase_index]), \
            transform(self.y_file[file_index]['recon_CS'][batch_index, phase_index])