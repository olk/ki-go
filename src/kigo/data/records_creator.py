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

import logging
import multiprocessing
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from pathlib import Path

from kigo.sgf import SGFGame
from kigo.board import Board, GameState, Move
from kigo.types import Player, Point
from kigo.encoders.base import get_encoder_by_name

class SGFTFRecordCreator(object):
    def __init__(self, encoder, sample_size):
        self._encoder = encoder
        self._sample_size = sample_size

    def _bytes_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _tensor_feature(self, values, dtype):
        values = tf.dtypes.cast(values, dtype)
        serialised_values = tf.io.serialize_tensor(values)
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialised_values.numpy()]))

    def _get_handicap(self, sgf):
        board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and 0 != sgf.get_handicap():
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(board, Player.white, None, move)
        return game_state, first_move_done

    def _number_of_moves(self, chunk):
        # determine the initial game state by applying all handicap stones
        num_moves = 0
        for sgf in chunk:
            # construct SGF from string
            sgf = SGFGame.from_string(sgf)
            game_state, first_move_done = self._get_handicap(sgf)
            for tpl in sgf.main_sequence_iter():
                color, move = tpl.get_move()
                if color is not None:
                    if first_move_done:
                        num_moves += 1
                    first_move_done = True
        return num_moves

    def _convert_to_tfrecord(self, chunk, dir_p, record_writer):
        # determine the shape of features and labels form the encoder
        shape = self._encoder.shape()
        for sgf in chunk:
            # construct SGF from string
            sgf = SGFGame.from_string(sgf)
            # determine the initial game state by applying all handicap stones
            game_state, first_move_done = self._get_handicap(sgf)
            # iterate over all moves in teh SGF (game) 
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        # get coordinates of this move
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        # pass 
                        move = Move.pass_turn()
                    if first_move_done and point is not None:
                        # encode tje current game state as feature
                        feature = self._encoder.encode(game_state)
                        ## reshape from (1, 19, 19) to (19, 19, 1)
                        #feature = np.moveaxis(feature, 0, -1)
                        # next move is the label for the this feature
                        label = self._encoder.encode_point(point)
                        label = to_categorical(label, 19 * 19)
                        # create a feature
                        feature = {
                                'feature': self._tensor_feature(feature, tf.int64),
                                'label': self._tensor_feature(label, tf.int64),
                        }
                        # create an example protocol buffer
                        ex = tf.train.Example(features=tf.train.Features(feature=feature))
                        # serialize to string and write on the file
                        record_writer.write(ex.SerializeToString())
                    # apply move to board and proceed with next one
                    game_state = game_state.apply_move(move)
                    first_move_done = True

    def _create(self, dir_p, idx, chunk):
        tfr_p = dir_p.joinpath('sgf_%s%d.tfr' % (self._encoder.name(), idx))
        print('creating %s' % tfr_p.name)
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(str(tfr_p), options=options) as record_writer:
            self._convert_to_tfrecord(chunk, dir_p, record_writer)

    def create(self, dir_p, sgfs):
        assert 0 < len(sgfs)
        if not dir_p.exists():
            dir_p.mkdir(parents=True)
        # partition data into lists with sample-size elements
        chunks = [sgfs[i:i+self._sample_size] for i in range(0, len(sgfs), self._sample_size)]
        # processing
        for i, chunk in enumerate(chunks):
            self._create(dir_p, i, chunk)
