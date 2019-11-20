'''
                    Copyright Oliver Kowalke 2018.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

from pathlib import Path


class SGFTFDatasetCreator(object):
    def __init__(self, dir_p, batch_size, train=True):
        files = [str(f) for f in dir_p.glob('*.tfr')]
        # read dataset from tfrecords
        ds = tf.data.TFRecordDataset(files, compression_type='GZIP')
        # decode/parse 
        self.dataset = ds.map(self._parse_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.n = self.dataset.reduce(0, lambda x, _: x + 1)

    def _parse_and_decode(self, serialized_ex):
        ex = tf.io.parse_single_example(
            serialized_ex,
            features={
              'feature': tf.io.FixedLenFeature([], tf.string),
              'label': tf.io.FixedLenFeature([], tf.string),
              }
            )
        feature = tf.io.parse_tensor(ex['feature'], tf.int64)
        feature = tf.reshape( feature, (19, 19, 1))
        label = tf.io.parse_tensor(ex['label'], tf.int64)
        label = tf.reshape( label, (19 * 19,))

        return feature, label
