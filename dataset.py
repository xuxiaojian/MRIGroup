import tensorflow as tf
import logging
import h5py


class MRIData(object):

    def __init__(self, root_path, scan_lines, mat_index, img_index: list, is_shuffle=True, is_liver_crop=False):
        self.liver_mask = tf.constant([
            [116, 244, 70, 198],
            [96, 224, 51, 179],
            [169, 297, 111, 239],
            [121, 249, 101, 229],
            [111, 239, 93, 221],
            [106, 234, 103, 231],
            [96, 224, 98, 226],
            [51, 179, 103, 231],
            [76, 204, 86, 214],
        ])  # range from healthy 1 to 9.

        self.is_liver_crop = is_liver_crop

        self.data = MRIDataH5(root_path=root_path, mat_index=mat_index, img_index=img_index, scan_lines=scan_lines)

        self.tf_dataset = self.get_dataset(self.data.generator, output_shape=(self.data.phase, self.data.width, self.data.height))
        if is_shuffle:
            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=self.data.batches())
        self.tf_dataset = self.tf_dataset.repeat()

        self.img_dataset = self.get_dataset(self.data.img_generator, output_shape=(self.data.phase, self.data.width, self.data.height)).batch(self.data.img_batches()).repeat(1)

    def get_dataset(self, generator_, output_shape):
        dataset = tf.data.Dataset.from_generator(generator=generator_, output_shapes=((), output_shape, output_shape), output_types=(tf.int32, tf.float32, tf.float32))
        dataset = dataset.map(self.map_fn)

        return dataset

    def map_fn(self, i, x, y):

        def transform(index_, input_):
            output_ = tf.transpose(input_, [0, 2, 1])
            output_ = tf.expand_dims(output_, -1)

            output_ -= tf.reduce_min(output_)
            output_ /= tf.reduce_max(output_)

            if self.is_liver_crop:
                width_up = self.liver_mask[index_][0]
                width_bottom = self.liver_mask[index_][1]
                height_up = self.liver_mask[index_][2]
                height_bottom = self.liver_mask[index_][3]

                output_ = output_[:, width_up:width_bottom, height_up:height_bottom, :]

            return output_

        return transform(i, x), transform(i, y)


class MRIDataH5(object):
    # Read H5 source File of MRI data

    def __init__(self, root_path, mat_index, img_index, scan_lines):
        import glob
        self.img_index = img_index
        self.mat_index = mat_index

        file_path = glob.glob(root_path + '*_*')
        file_path.sort()

        self.x_h5file = []
        self.y_h5file = []
        for i in mat_index:
            logging.root.info("[MRIDataH5] Get h5 file with index [%d]." % i + "Path: " + file_path[i])
            self.x_h5file.append(h5py.File(file_path[i] + '/MCNUFFT_' + scan_lines + '.h5', 'r'))
            self.y_h5file.append(h5py.File(file_path[i] + '/CS_2000' + '.h5', 'r'))

        _, self.phase, self.width, self.height = self.x_h5file[0]['recon_MCNUFFT'].shape

    def generator(self):
        for i in range(self.x_h5file.__len__()):
            batch_file, _, _, _ = self.x_h5file[i]['recon_MCNUFFT'].shape
            for batch in range(batch_file):
                yield self.mat_index[i], self.x_h5file[i]['recon_MCNUFFT'][batch], self.y_h5file[i]['recon_CS'][batch]

    def batches(self):
        output_ = 0
        for i in range(self.x_h5file.__len__()):
            batch_file, _, _, _ = self.x_h5file[i]['recon_MCNUFFT'].shape
            output_ += batch_file

        return output_

    def img_generator(self):
        for i in self.img_index:
            yield self.mat_index[0], self.x_h5file[0]['recon_MCNUFFT'][i], self.y_h5file[0]['recon_CS'][i]

    def img_batches(self):

        return self.img_index.__len__()
