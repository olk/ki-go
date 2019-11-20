# tag::data_generator[]
import glob
import numpy as np
from tensorflow.keras.utils import to_categorical


class DataGenerator:
    def __init__(self, data_directory, data_type):
        self.data_directory = data_directory
        self.data_type = data_type
        self.num_samples = None

    def get_num_samples(self, batch_size=128, num_classes=19 * 19):
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size, num_classes=num_classes):
                self.num_samples += X.shape[0]
            return self.num_samples

    def _generate(self, batch_size, num_classes):
        for feature_file in self.data_directory.glob('*' + self.data_type + '_features.npy'):
            label_file = feature_file.parent.joinpath(feature_file.stem.replace('features', 'labels')).with_suffix('.npy')
            assert feature_file.exists()
            assert label_file.exists()
            x = np.load(feature_file)
            y = np.load(label_file)
            x = x.astype('float32')
            y = to_categorical(y.astype(int), num_classes)
            while x.shape[0] >= batch_size:
                x_batch, x = x[:batch_size], x[batch_size:]
                y_batch, y = y[:batch_size], y[batch_size:]
                yield x_batch, y_batch


    def generate(self, batch_size=128, num_classes=19 * 19):
        while True:
            for item in self._generate(batch_size, num_classes):
                yield item
