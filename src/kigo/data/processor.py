import glob
import gzip
import numpy as np
import os.path
import shutil
import tarfile

from tensorflow.keras.utils import to_categorical
from pathlib import Path
from kigo.sgf.sgf import SGFGame
from kigo.board import Board, GameState, Move
from kigo.types import Player, Point
from kigo.encoders.base import get_encoder_by_name

from kigo.data.index_processor import KGSIndex
from kigo.data.generator import DataGenerator


class GoDataProcessor:
    def __init__(self, data_p, encoder='oneplane'):
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_p = data_p
        index = KGSIndex(data_p=self.data_p)
        index.download_files()
        self._train, self._val, self._test = self._split_data(0.8, 0.1)

    def _split_data(self, train_frac, val_frac):
        files = list(self.data_p.glob('*.tar.gz'))
        # create array of indices
        # each index represents one spreadsheet (e.g. jpg)
        indices = np.arange(0, len(files))
        # split indices in training/validation/testing subsets
        train_indices, test_indices, val_indices = np.split(indices, [int(train_frac * len(indices)), int((1 - val_frac) * len(indices))])
        # split jpgs according to the indices
        train = files[train_indices[0]:train_indices[-1]+1]
        test = files[test_indices[0]:test_indices[-1]+1]
        val = files[val_indices[0]:val_indices[-1]+1]
        return train, val, test

    def load_go_data(self, data_type='train'):
        if 'train' == data_type:
            data = self._train
        elif 'test' == data_type:
            data = self._test
        elif 'val' == data_type:
            data = self._val
        else:
            raise ValueError(data_type + " is not a valid data type, choose from 'train' or 'test'")

        zip_names = set()
        indices_by_zip_name = {}
        for index, filename in enumerate(data):
            zip_names.add(filename)  # <5>
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)  # <6>
        for zip_name in zip_names:
            data_file_name = zip_name.with_name(Path(zip_name.stem).stem + data_type)
            if not data_file_name.is_file():
                self.process_zip(zip_name, data_file_name, indices_by_zip_name[zip_name])  # <7>

        generator = DataGenerator(self.data_p, data_type)
        return generator


    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(zip_file_name)  # <1>

        tar_file = zip_file_name.with_suffix('')  # <2>
        this_tar = open(tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)  # <3>
        this_tar.close()
        return tar_file

    def process_zip(self, zip_file_name, data_file_name, game_list):
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(tar_file)
        name_list = zip_file.getnames()
        total_examples = self.num_total_examples(zip_file, game_list, name_list)  # <1>

        shape = self.encoder.shape()  # <2>
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))

        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = SGFGame.from_string(sgf_content)  # <3>

            game_state, first_move_done = self.get_handicap(sgf)  # <4>

            for item in sgf.main_sequence_iter():  # <5>
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:  # <6>
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()  # <7>
                    if first_move_done and point is not None:
                        features[counter] = self.encoder.encode(game_state)  # <8>
                        labels[counter] = self.encoder.encode_point(point)  # <9>
                        counter += 1
                    game_state = game_state.apply_move(move)  # <10>
                    first_move_done = True

        feature_file = data_file_name.with_name(data_file_name.stem + '_features')
        label_file = data_file_name.with_name(data_file_name.stem + '_labels')

        np.save(feature_file, features)
        np.save(label_file, labels)

    @staticmethod
    def get_handicap(sgf):
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = SGFGame.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples
