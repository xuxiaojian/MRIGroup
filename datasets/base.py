import tensorflow as tf


class DatasetBase(object):
    def __init__(self, is_shuffle, indexes_length, x_shape, y_shape):

        self.x_shape = x_shape
        self.y_shape = y_shape

        self.tf_dataset = tf.data.Dataset.from_generator(
            generator=self.dataset_generator, output_shapes=indexes_length, output_types=tf.int32)
        if is_shuffle:
            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=self.get_dataset_len())

        self.tf_dataset = self.tf_dataset.map(self.get_tf_map_fn).repeat()

        self.tf_sample = tf.data.Dataset.from_generator(
            generator=self.sample_generator, output_shapes=indexes_length, output_types=tf.int32). \
            map(self.get_tf_map_fn).repeat().batch(self.get_sample_len())

    def get_tf_map_fn(self, indexes):
        x, y = tf.py_function(self.read_data, [indexes], [tf.float32, tf.float32])
        x.set_shape(self.x_shape)
        y.set_shape(self.y_shape)
        return x, y

    def get_dataset_len(self):
        pass

    def get_sample_len(self):
        pass

    def dataset_generator(self):
        pass

    def sample_generator(self):
        pass

    def read_data(self, indexes) -> tuple:
        pass
