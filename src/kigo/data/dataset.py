import mxnet as mx
import numpy as np
import os
import random
import shutil
import six
import tarfile

from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from threading import Event, Thread
from urllib.request import urlopen, urlretrieve

from kigo.board import Board, GameState, Move
from kigo.sgf import SGFGame
from kigo.types import Player, Point


_URL='http://u-go.net/gamerecords/'


def _decode(npz_p):
    npz = np.load(str(npz_p))
    l = npz['l']
    d = npz['d']
    assert 0 < len(l), 'label has zero length'
    assert len(l) == len(d), 'length of label differs from data'
    return d, l


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
        ## determine winner
        #winner = sgf.get_winner()
        #if winner is None:
        #    print('no winner: %s' % sgf_p.name)
        #    return
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
                if first_move_done and point is not None:
                # use only winner's moves
                #if first_move_done and point is not None and winner == color:
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

    def train_iter(self, batch_size, max_workers=4, num_games=None, num_segments=1, num_prefetch=1, shuffle=False):
        npzs_p = list(self.processed_p.glob(('train-%s-*.npz') % self.encoder.name()))
        return SGFDataIter(npzs_p, batch_size, max_workers, num_games, num_segments, num_prefetch, self.encoder.shape(), shuffle)

    def val_iter(self, batch_size, max_workers=4, num_games=None, num_segments=1, num_prefetch=1, shuffle=False):
        npzs_p = list(self.processed_p.glob(('val-%s-*.npz') % self.encoder.name()))
        return SGFDataIter(npzs_p, batch_size, max_workers, num_games, num_segments, num_prefetch, self.encoder.shape(), shuffle)

    def test_iter(self, batch_size, max_workers=4, num_games=None, num_segments=1, num_prefetch=1, shuffle=False):
        npzs_p = list(self.processed_p.glob(('test-%s-*.npz') % self.encoder.name()))
        return SGFDataIter(npzs_p, batch_size, max_workers, num_games, num_segments, num_prefetch, self.encoder.shape(), shuffle)

    def train_nditer(self, batch_size, num_games=None, shuffle=False):
        npzs_p = list(self.processed_p.glob(('train-%s-*.npz') % self.encoder.name()))
        if num_games is not None:
            npzs_p = random.sample(npzs_p, num_games)
        data = []
        label = []
        with ProcessPoolExecutor() as pool:
            it = pool.map(_decode, npzs_p, chunksize=100)
            for d, l in it:
                data.extend(d)
                label.extend(l)
        return mx.io.NDArrayIter(mx.nd.array(data), mx.nd.array(label), batch_size=batch_size, shuffle=shuffle)


class SGFDataIter(mx.io.DataIter):
    def __init__(self, npz, batch_size, max_workers, num_games, num_segments, num_prefetch, shape, shuffle):
        super().__init__(batch_size)
        self._provide_data = [mx.io.DataDesc('data', (batch_size,) + shape, layout='NCHW')]
        self._provide_label = [mx.io.DataDesc('label', (batch_size,), layout='NCHW')]

        if num_games is not None:
            npz = random.sample(npz, num_games)
        self._idx = 0
        self._fetch_idx = 0
        self._npz = npz
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._max_workers = max_workers
        self._num_segments = num_segments
        self._num_prefetch = num_prefetch
        self._segment_length = ceil(len(self._npz)/self._num_segments)
        self._iters = [None]*self._num_segments
        self._data_ready = [Event() for i in range(self._num_segments)]
        self._data_taken = [Event() for i in range(self._num_segments)]
        self._running = True

        def _do_fetch(self, idx):
            while True:
                self._data_taken[idx].wait()
                self._data_taken[idx].clear()
                if not self._running:
                    break;
                data = []
                label = []
                begin = idx * self._segment_length
                end = begin + self._segment_length

                for i in range(begin, end):
                    npz = np.load(str(self._npz[i]))
                    l = npz['l']
                    d = npz['d']
                    assert 0 < len(l), 'label has zero length'
                    assert len(l) == len(d), 'length of label differs from data'
                    data.extend(d)
                    label.extend(l)

                self._iters[idx] = mx.io.NDArrayIter(
                    mx.nd.array(data),
                    mx.nd.array(label),
                    batch_size=self._batch_size,
                    shuffle=self._shuffle)
                # delete arrays because of waiting in _data_taken.wait()
                del data, label
                self._data_ready[idx].set()

        self._worker = [Thread(target=_do_fetch, args=[self, i]) for i in range(self._num_segments)]
        for w in self._worker:
            w.setDaemon(True)
            w.start()

        print('%d SGF files in %d segments each containing upto %d SGF files' % (len(self._npz), self._num_segments, self._segment_length))

    def _fetch_next(self):
        if self._fetch_idx < self._num_segments:
            self._data_taken[self._fetch_idx].set()
            self._fetch_idx += 1

    def __del__(self):
        self._running = False
        for ev in self._data_taken:
            ev.set()
        for w in self._worker:
            w.join()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return next(self._iters[self._idx])
            except StopIteration:
                self._iters[self._idx] = None
                self._idx += 1
                if self._idx == self._num_segments:
                    raise
                self._fetch_next()
                self._data_ready[self._idx].wait()
                self._data_ready[self._idx].clear()

    def reset(self):
        self._idx = 0
        self._fetch_idx = 0
        for _ in range(self._num_prefetch):
            self._fetch_next()
        self._data_ready[self._idx].wait()
        self._data_ready[self._idx].clear()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
