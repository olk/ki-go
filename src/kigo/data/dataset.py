import mxnet as mx
import numpy as np
import os
import random
import shutil
import six
import tarfile
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from kigo.board import Board, GameState, Move
from kigo.sgf import SGFGame
from kigo.types import Player, Point


_URL='http://u-go.net/gamerecords/'


class SGFDatasetBuilder:
    def __init__(self, dir_p, encoder, train_frac=0.8, val_frac=0.1):
        self.root_p = dir_p.joinpath('sgf_dataset')
        self.downloads_p = self.root_p.joinpath('downloads')
        self.processed_p = self.root_p.joinpath('dataset_builder')
        self.encoder = encoder
        self.train_frac = train_frac
        self.val_frac = val_frac

    def _get_members(self, tar):
        for info in tar.getmembers():
            info.name = Path(info.name).name
            yield info

    def _download_and_extract(self, url_and_target):
        (url, target_p) = url_and_target
        urlretrieve(url, str(target_p))
        with tarfile.open(str(target_p)) as tar:
            tar.extractall(target_p.parent, self._get_members(tar))
        target_p.unlink()

    def _get_game_index(self):
        with urlopen(_URL) as fp:
            index_contents = six.text_type(fp.read())
        return index_contents

    def _download(self):
        # download
        index = self._get_game_index()
        split_page = [item for item in index.split('<a href="') if item.startswith("https://")]
        urls = []
        for itm in split_page:
            url = itm.split('">Download')[0]
            if url.endswith('.tar.gz'):
                filename = os.path.basename(url)
                urls.append((url, self.downloads_p.joinpath(filename)))
        print('downloading %d SGF archives' % len(urls))
        with ProcessPoolExecutor() as pool:
            pool.map(self._download_and_extract, urls, chunksize=10)
        for itm in self.downloads_p.iterdir():
            if itm.is_dir():
                shutil.rmtree(itm)

    def _get_handicap(self, sgf):
        board = Board(19, 19)
        first_move_done = False
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None:
            point = None
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    point = Point(row + 1, col + 1)
                    board.place_stone(Player.black, point)
            first_move_done = True
            if point is not None:
                game_state = GameState(board, Player.white, None, Move.play(point))
        return game_state, first_move_done

    def _encode_and_persist(self, sgf_p):
        sgf = SGFGame.from_string(sgf_p.read_text())
        # determine winner
        winner = sgf.get_winner()
        if winner is None:
            print('no winner: %s' % sgf_p.name)
            return
        # determine the initial game state by applying all handicap stones
        game_state, first_move_done = self._get_handicap(sgf)
        label = []
        data = []
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
                        print('invalid move: %s' % sgf_p.name)
                        return
                else:
                    # pass 
                    move = Move.pass_turn()
                # use only winner's moves
                if first_move_done and point is not None and winner == color:
                    # encode the current game state as feature
                    d = self.encoder.encode(game_state)
                    # next move is the label for the this feature
                    l = self.encoder.encode_point(point)
                    data.append(d)
                    label.append(l)
                # apply move to board and proceed with next one
                game_state = game_state.apply_move(move)
                first_move_done = True
        # create numpy compressed file
        size = len(data)
        if 0 == size:
            print('empty: %s' % sgf_p.name)
            return
        assert len(label) == size, 'label with invalid size'
        assert len(data) == size, 'data with invalid size'
        npz_p = self.processed_p.joinpath('%s-%s-%d' % (self.encoder.name(), sgf_p.stem, size))
        label = np.array(label, dtype=np.int)
        data = np.array(data, dtype=np.int)
        np.savez_compressed(str(npz_p), d=data, l=label)

    def _split_files(self, npzs_p):
        # create array of indices
        # each index represents one spreadsheet (e.g. jpg)
        indices = np.arange(0, len(npzs_p))
        # split indices in training/validation/testing subsets
        train_indices, test_indices, val_indices = np.split(indices, [int(self.train_frac * len(indices)), int((1 - self.val_frac) * len(indices))])
        # split numpy compressed files according to the indices
        train_npzs_p = npzs_p[train_indices[0]:train_indices[-1]+1]
        test_npzs_p = npzs_p[test_indices[0]:test_indices[-1]+1]
        val_npzs_p = npzs_p[val_indices[0]:val_indices[-1]+1]
        return train_npzs_p, val_npzs_p, test_npzs_p

    def _prepare(self):
        # encode
        sgfs_p = list(self.downloads_p.glob('*.sgf'))
        print('encoding %d SGF files' % len(sgfs_p))
        with ProcessPoolExecutor() as pool:
            pool.map(self._encode_and_persist, sgfs_p, chunksize=100)
        # split and rename
        npzs_p = list(self.processed_p.glob(self.encoder.name() + '-*.npz'))
        print('created %d numpy compressed files' % len(npzs_p))
        train_npzs_p, val_npzs_p, test_npzs_p = self._split_files(npzs_p)
        for npz_p in train_npzs_p:
            new_npz_p = npz_p.parent.joinpath('train-' + npz_p.stem).with_suffix('.npz')
            shutil.move(npz_p, new_npz_p)
        for npz_p in val_npzs_p:
            new_npz_p = npz_p.parent.joinpath('val-' + npz_p.stem).with_suffix('.npz')
            shutil.move(npz_p, new_npz_p)
        for npz_p in test_npzs_p:
            new_npz_p = npz_p.parent.joinpath('test-' + npz_p.stem).with_suffix('.npz')
            shutil.move(npz_p, new_npz_p)

    def download_and_prepare(self, force_download=False, force_prepare=False):
        if force_download:
            shutil.rmtree(self.downloads_p, ignore_errors=True)
            shutil.rmtree(self.processed_p, ignore_errors=True)
        if not self.downloads_p.exists():
            self.downloads_p.mkdir(parents=True)
        assert self.downloads_p.is_dir()
        if 0 == len(os.listdir(str(self.downloads_p))):
            self._download()
        if force_prepare:
            shutil.rmtree(self.processed_p, ignore_errors=True)
        if not self.processed_p.exists():
            self.processed_p.mkdir(parents=True)
        assert self.processed_p.is_dir()
        if 0 == len(os.listdir(str(self.processed_p))):
            self._prepare()

    def train_dataset(self, batch_size, max_worker=4, shuffle=False, factor=5):
        npzs_p = list(self.processed_p.glob(('train-%s-*.npz') % self.encoder.name()))
        print(len(npzs_p))
        if shuffle:
            random.shuffle(npzs_p)
        return SGFDataIter(npzs_p, batch_size, max_worker, self.encoder, shuffle, factor)

    def val_dataset(self, batch_size, max_worker=4, shuffle=False, factor=5):
        npzs_p = list(self.processed_p.glob(('val-%s-*.npz') % self.encoder.name()))
        if shuffle:
            random.shuffle(npzs_p)
        return SGFDataIter(npzs_p, batch_size, max_worker, self.encoder, shuffle, factor)

    def test_dataset(self, batch_size, max_worker=4, shuffle=False, factor=5):
        npzs_p = list(self.processed_p.glob(('test-%s-*.npz') % self.encoder.name()))
        if shuffle:
            random.shuffle(npzs_p)
        return SGFDataIter(npzs_p, batch_size, max_worker, self.encoder, shuffle, factor)


_batch_size = None
_dataset = None
_decoder = None
_shape = None
_z = None
_idx = None
_max_idx = None
_shuffle = None
_npzs_per_batch = None
_results = None
_factor = None


def _init_decoder(dataset, shape, shuffle):
    global _dataset
    global _shape
    global _shuffle
    _dataset = dataset
    _shape = shape
    _shuffle = shuffle


def _decode_fn(idx):
    global _dataset
    global _shape
    global _shuffle
    assert _dataset is not None
    assert _shape is not None
    assert _shuffle is not None
    npz = np.load(_dataset[idx])
    label = npz['l']
    data = npz['d']
    assert 0 < len(label), 'label has invalid size'
    assert len(label) == len(data), 'label not of same size as data'
    z = list(zip(data, label))
    if _shuffle:
        random.shuffle(z)
    return z


def _init_collector(max_worker, batch_size, dataset, shape, shuffle, factor):
    global _batch_size
    global _decoder
    global _z
    global _idx
    global _max_idx
    global _npzs_per_batch
    global _results
    global _factor
    _batch_size = batch_size
    _z = []
    _idx = 0
    _max_idx = len(dataset)
    _results = []
    _factor = factor
    # assumption: each numpy compressed file contains 100 moves at average
    _npzs_per_batch = ceil(_batch_size / 100)
    _decoder = ProcessPoolExecutor(max_workers=max_worker, initializer=_init_decoder, initargs=(dataset, shape, shuffle))


def _batchify_fn(reinit):
    global _batch_size
    global _decoder
    global _z
    global _idx
    global _max_idx
    global _regs_per_batch
    global _results
    global _factor
    assert _batch_size is not None
    assert _decoder is not None
    assert _z is not None
    assert _idx is not None
    assert _max_idx is not None
    assert _npzs_per_batch is not None
    assert _factor is not None
    # reset
    if reinit:
        _z = [] # del _z?
        _idx = 0
        while len(_z) < _factor * _batch_size and _idx < _max_idx:
            # request data
            new_idx = min(_idx + _npzs_per_batch, _max_idx)
            results = [_decoder.submit(_decode_fn, idx) for idx in range(_idx, new_idx)]
            _idx = new_idx
            # fetch data
            for f in as_completed(results):
                _z += f.result()
            # request data
            new_idx = min(_idx + _npzs_per_batch, _max_idx)
            _results = [_decoder.submit(_decode_fn, idx) for idx in range(_idx, new_idx)]
            _idx = new_idx
    else:
        # fetch data
        for f in as_completed(_results):
            _z += f.result()
        _results = []
        if len(_z) < _factor * _batch_size and _idx < _max_idx:
            # request data
            new_idx = min(_idx + _npzs_per_batch, _max_idx)
            _results = [_decoder.submit(_decode_fn, idx) for idx in range(_idx, new_idx)]
            _idx = new_idx
    # no more data
    if 0 == len(_z) and _idx == _max_idx:
        return None
    # create batch
    z, _z = _z[:_batch_size], _z[_batch_size:]
    pad = 0
    if len(z) < _batch_size:
        pad = _batch_size - len(z)
        # pad with first element
        # padded elelments will be ignored
        z = z + [z[0]*pad]
    d, l = zip(*z)
    return mx.io.DataBatch([d],[l], pad)


class SGFDataIter(mx.io.DataIter):
    def __init__(self, dataset, batch_size, max_worker, encoder, shuffle, factor):
        self._provide_data = [mx.io.DataDesc('data', (batch_size,) + encoder.shape(), layout='NCHW')]
        self._provide_label = [mx.io.DataDesc('label', (batch_size,), layout='NCHW')]
        self._batchifier = ThreadPoolExecutor(max_workers=1, initializer=_init_collector, initargs=(max_worker, batch_size, dataset, encoder.shape(), shuffle, factor))
        self._fut = None

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._fut.result()
        if batch is None:
            raise StopIteration
        # request next batch
        self._fut = self._batchifier.submit(_batchify_fn, False)
        return batch

    def reset(self):
        if self._fut is not None:
            self._fut.cancel()
        self._fut = self._batchifier.submit(_batchify_fn, True)

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
