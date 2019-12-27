# tag::data_generator[]
import glob
import numpy as np

from keras import backend as K
from keras.utils import Sequence
from keras.utils import to_categorical


class DataGenerator(Sequence):
    def __init__(self, data_directory, batch_size):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.n = None

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(self.n/self.batch_size))

    def __getitem__(self, idx):
        # generate one batch of data
        # generate indexes of the batch
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def __data_generation(self, indexes):
        # generates data containing batch-size samples
        assert 'channels_first' == K.image_data_format()
        for feature_file in self.data_directory.glob('*' + '_features.npy'):
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
                return x_batch, y_batch

    def on_epoch_end(self):
        # updates indexes after each epoch
        self.indexes = np.arange(self.n)
        np.random.shuffle(self.indexes)
