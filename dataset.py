import tensorflow as tf
import numpy as np
import logging
import h5py
import glob


class MRIData(object):
    def __init__(self, file_index: list,
                 scan_lines: str = '400',
                 root_path: str = '/export/project/gan.weijie/dataset/mri_source/',
                 sample_index: list = (16, 21, 27, 30),
                 is_temporal: bool = True,
                 time_step: int = 3,
                 is_shuffle: bool = True,
                 ):

        """
        Key variables in this object will be: (1) dataset: main variable and (2) sample: certain images for visualization.

        :param root_path: The root-level path of MRI data. Example of mri data file: root_path/healthy_01/*.
        :param scan_lines: Scan lines of MCNUFFT for noised input in training, ground-truth will always be 2000 line CS.
        :param file_index: Load certain index of data.
        :param sample_index: Load certain index of image in the **first** index of data.
        :param is_temporal: If True, the dimension of data will be None * temporal * phase * width * height * channel.
        """
        logging.root.info("[MRIData Object]")

        # Read Source H5 File Pointer
        self.x_file = []; self.y_file = []
        self.sample_index = sample_index

        self.is_temporal = is_temporal
        self.time_step = time_step  # valid if is_temporal=True

        x_file_path = glob.glob(root_path + '*/MCNUFFT_' + scan_lines + '*.h5'); x_file_path.sort()
        y_file_path = glob.glob(root_path + '*/CS_2000' + '*.h5'); y_file_path.sort()

        for i in file_index:
            logging.root.info("Loading H5File Pointer in Path: " + x_file_path[i] + '.')
            self.x_file.append(h5py.File(x_file_path[i], 'r'))
            self.y_file.append(h5py.File(y_file_path[i], 'r'))

        _, self.phase, self.width, self.height = self.x_file[0]['recon_MCNUFFT'].shape
        self.channel = 1

        # Construct dataset
        if is_temporal:
            output_shape = (self.time_step, self.phase, self.width, self.height)
        else:
            output_shape = (self.phase, self.width, self.height)

        self.tf_dataset = tf.data.Dataset.from_generator(generator=self.dataset_generator, output_shapes=(output_shape, output_shape),
                                                         output_types=(tf.float32, tf.float32)).map(self.preprocess_fn)
        if is_shuffle:
            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=self.dataset_len())
        self.tf_dataset = self.tf_dataset.repeat()

        self.tf_sample = tf.data.Dataset.from_generator(generator=self.sample_generator, output_shapes=(output_shape, output_shape),
                                                        output_types=(tf.float32, tf.float32)).map(self.preprocess_fn).repeat().batch(self.sample_len())

    def dataset_len(self):
        output_len = 0
        for i in self.x_file:
            output_len += i['recon_MCNUFFT'].shape[0]
        return output_len

    def sample_len(self):
        return len(self.sample_index)

    def dataset_generator(self):
        for file_index in range(len(self.x_file)):
            batches = self.x_file[file_index]['recon_MCNUFFT'].shape[0]

            if self.is_temporal:
                for batch_index in range(batches):
                    temporal_index = np.array([i for i in range(self.time_step)]) + batch_index - int(self.time_step / 2)
                    if batch_index >= (batches - int(self.time_step / 2)):  # due to slice rule of h5py that the indexes must be increasing
                        temporal_index -= batches
                    temporal_index = temporal_index.tolist()

                    yield self.x_file[file_index]['recon_MCNUFFT'][temporal_index],\
                        self.y_file[file_index]['recon_CS'][temporal_index]
            else:
                for batch_index in range(batches):
                    yield self.x_file[file_index]['recon_MCNUFFT'][batch_index], \
                          self.y_file[file_index]['recon_CS'][batch_index]

    def sample_generator(self):
        for i in self.sample_index:
            if self.is_temporal:
                batches = self.x_file[0]['recon_MCNUFFT'].shape[0]
                temporal_index = np.array([i for i in range(self.time_step)]) + i - int(self.time_step / 2)
                if i >= (batches - int(
                        self.time_step / 2)):  # due to slice rule of h5py that the indexes must be increasing
                    temporal_index -= batches
                temporal_index = temporal_index.tolist()

                yield self.x_file[0]['recon_MCNUFFT'][temporal_index], self.y_file[0]['recon_CS'][temporal_index]
            else:
                yield self.x_file[0]['recon_MCNUFFT'][i], self.y_file[0]['recon_CS'][i]

    def preprocess_fn(self, x, y):
        def transform(input_):
            output_ = tf.transpose(input_, [0, 2, 1])
            output_ -= tf.reduce_min(output_)
            output_ /= tf.reduce_max(output_)

            return output_

        if self.is_temporal:
            return tf.expand_dims(tf.map_fn(transform, x), -1), tf.expand_dims(tf.map_fn(transform, y), -1)

        else:
            return tf.expand_dims(transform(x), -1), tf.expand_dims(transform(y), -1)
