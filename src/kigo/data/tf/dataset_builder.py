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
import os
import six
import ssl
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from enum import Enum
from pathlib import Path
from random import sample, shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from urllib.request import urlopen

from kigo.sgf import SGFGame
from kigo.board import Board, GameState, Move
from kigo.types import Player, Point
from kigo.encoders.base import get_encoder_by_name


_DESCRIPTION = ('')
_SUPERVISED_KEYS = ('feature', 'label')
_URL = 'https://u-go.net/gamerecords/'

checksum_dir = os.path.join(os.path.dirname(__file__),'url_checksums/')
checksum_dir = os.path.normpath(checksum_dir)
tfds.download.add_checksums_dir(checksum_dir)


def _get_handicap(sgf):
    board = Board(19, 19)
    first_move_done = False
    game_state = GameState.new_game(19)
    if sgf.get_handicap() is not None and 0 != sgf.get_handicap():
        point = None
        for setup in sgf.get_root().get_setup_stones():
            for move in setup:
                row, col = move
                point = Point(row + 1, col + 1)
                board.place_stone(Player.black, point)
        first_move_done = True
        if point is None:
            return None
        game_state = GameState(board, Player.white, None, Move.play(point))
    return game_state, first_move_done

def _process_sgf(sgf_p, encoder, shape):
    label_features = []
    # construct SGF from string
    sgf = SGFGame.from_string(sgf_p.read_text())
    # determine the initial game state by applying all handicap stones
    tpl = _get_handicap(sgf)
    if tpl is None:
        print('invalid handicap in SGF %s: excluded' % (sgf_p.name))
        return None
    game_state, first_move_done = tpl
    # iterate over all moves in the SGF (game) 
    for item in sgf.main_sequence_iter():
        color, move_tuple = item.get_move()
        point = None
        if color is not None:
            if move_tuple is not None:
                # get coordinates of this move
                row, col = move_tuple
                point = Point(row + 1, col + 1)
                move = Move.play(point)
                # allow only valid moves
                if not game_state.is_valid_move(move):
                    print('invalid move in SGF %s: excluded' % (sgf_p.name))
                    return None
            else:
                # pass 
                move = Move.pass_turn()
            if first_move_done and point is not None:
                # encode the current game state as feature
                feature = encoder.encode(game_state)
                feature = feature.astype('int32')
                ## reshape from (1, 19, 19) to (19, 19, 1)
                #feature = np.moveaxis(feature, 0, -1)
                # next move is the label for the this feature
                label = encoder.encode_point(point)
                label = to_categorical(label, 19 * 19, dtype='int32')
                label_features.append((label, feature))
            # apply move to board and proceed with next one
            game_state = game_state.apply_move(move)
            first_move_done = True
    return label_features


class SGFDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """ SGF Go dataset """
    VERSION = tfds.core.Version('0.1.0')

    def __init__(self, encoder, train_frac=0.8, val_frac=0.1, sgf_files=0):
        self._encoder = encoder
        self._train_frac = train_frac
        self._val_frac = val_frac
        self._sgf_files = sgf_files
        super(SGFDatasetBuilder, self).__init__()

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'feature': tfds.features.Tensor(shape=self._encoder.shape(), dtype=tf.int32),
                'label': tfds.features.Tensor(shape=(19*19,), dtype=tf.int32),
            }),
            supervised_keys=_SUPERVISED_KEYS,
        )

    def _split_data(self, sgfs):
        # create array of indices
        # each index represents one spreadsheet (e.g. jpg)
        indices = np.arange(0, len(sgfs))
        # split indices in training/validation/testing subsets
        train_indices, test_indices, val_indices = np.split(indices, [int(self._train_frac * len(indices)), int((1 - self._val_frac) * len(indices))])
        # split sgfs according to the indices
        train_sgfs = sgfs[train_indices[0]:train_indices[-1]+1]
        test_sgfs = sgfs[test_indices[0]:test_indices[-1]+1]
        val_sgfs = sgfs[val_indices[0]:val_indices[-1]+1]
        return train_sgfs, val_sgfs, test_sgfs

    def _split_generators(self, dl_manager):
        # get list of archives
        context = ssl._create_unverified_context()
        urls = []
        with urlopen(_URL, context=context) as fp:
            index_contents = six.text_type(fp.read())
        split_page = [item for item in index_contents.split('<a href="') if item.startswith("https://")]
        for item in split_page:
            download_url = item.split('">Download')[0]
            if download_url.endswith('.tar.bz2'):
                urls.append(download_url)
        # download and extract SGF files
        paths = dl_manager.download_and_extract(urls)
        # get SGF files
        sgfs_p = [x for y in [list(Path(path).glob('**/*.sgf')) for path in paths] for x in y]
        shuffle(sgfs_p)
        if 0 < self._sgf_files:
            sgfs_p = sample(sgfs_p, self._sgf_files)
        print('%d SGF files' % len(sgfs_p))

        # shuffle SGF files from different years
        # split SGF files for training, validation and test
        train_sgfs_p, val_sgfs_p, test_sgfs_p = self._split_data(sgfs_p)
        # return dataset
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=1,
                gen_kwargs={'sgfs_p': train_sgfs_p}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=1,
                gen_kwargs={'sgfs_p': val_sgfs_p}),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=1,
                gen_kwargs={'sgfs_p': test_sgfs_p}),
        ]

    def _generate_examples(self, sgfs_p):
        # read and process SGF files parallel
        shape = self._encoder.shape()
        id = 0
        for sgf_p in sgfs_p:
            label_features = _process_sgf(sgf_p, self._encoder, shape)
            if label_features is not None:
                for label, feature in label_features:
                    yield id, {
                            'feature': feature,
                            'label': label,
                    }
                    id += 1

    @classmethod
    def size(cls, dataset):
        return dataset.reduce(0, lambda x, _: x + 1)
