import multiprocessing
import mxnet as mx
import numpy as np
import os
import random
import shutil
import six
import tarfile

from functools import partial
from kigo.board import Board, GameState, Move
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.sgf import SGFGame
from kigo.types import Player, Point
from math import ceil
from pathlib import Path
from urllib.request import urlopen, urlretrieve

_URL='http://u-go.net/gamerecords/'

class SGFDatasetBuilder:
    def __init__(self, dir_p, encoder=SevenPlaneEncoder((19, 19)), train_frac=0.8, val_frac=0.1):
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
        print('downloading ' + target_p.name)
        urlretrieve(url, str(target_p))
        with tarfile.open(str(target_p)) as tar:
            print('extracting ' + target_p.name)
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

        size = len(urls)
        pool = multiprocessing.Pool(processes=size)
        try:
            it = pool.imap(self._download_and_extract, urls)
            for _ in it:
                pass
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(1)

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
        print('encoding: %s' % sgf_p.name)
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
        # create recordio file
        rec_p = self.processed_p.joinpath(self.encoder.name() + '-' + sgf_p.stem).with_suffix('.rec')
        size = len(data)
        assert len(label) == size
        record = mx.recordio.MXRecordIO(str(rec_p), 'w')
        record.write(str(size).encode('utf-8'))
        for idx, (l, d) in enumerate(zip(label, data)):
            hdr = mx.recordio.IRHeader(0, l, idx, 0)
            s = d.tobytes()
            p = mx.recordio.pack(hdr, s)
            record.write(p)
        record.close()

    def _split_files(self, recs_p):
        # create array of indices
        # each index represents one spreadsheet (e.g. jpg)
        indices = np.arange(0, len(recs_p))
        # split indices in training/validation/testing subsets
        train_indices, test_indices, val_indices = np.split(indices, [int(self.train_frac * len(indices)), int((1 - self.val_frac) * len(indices))])
        # split records according to the indices
        train_recs_p = recs_p[train_indices[0]:train_indices[-1]+1]
        test_recs_p = recs_p[test_indices[0]:test_indices[-1]+1]
        val_recs_p = recs_p[val_indices[0]:val_indices[-1]+1]
        return train_recs_p, val_recs_p, test_recs_p

    def _prepare(self):
        # encode
        sgfs_p = list(self.downloads_p.glob('*.sgf'))
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        try:
            it = pool.imap(self._encode_and_persist, sgfs_p)
            for _ in it:
                pass
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(1)
        # split and rename
        recs_p = list(self.processed_p.glob(self.encoder.name() + '-*.rec'))
        train_recs_p, val_recs_p, test_recs_p = self._split_files(recs_p)
        for rec_p in train_recs_p:
            new_rec_p = rec_p.parent.joinpath('train-' + rec_p.stem).with_suffix('.rec')
            shutil.move(rec_p, new_rec_p)
        for rec_p in val_recs_p:
            new_rec_p = rec_p.parent.joinpath('val-' + rec_p.stem).with_suffix('.rec')
            shutil.move(rec_p, new_rec_p)
        for rec_p in test_recs_p:
            new_rec_p = rec_p.parent.joinpath('test-' + rec_p.stem).with_suffix('.rec')
            shutil.move(rec_p, new_rec_p)

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

    def get_train_iter(self, batch_size, shuffle=False, num_games=None):
        recs_p = list(self.processed_p.glob(('train-%s-*.rec') % self.encoder.name()))
        itr = SGFIter(recs_p, batch_size, self.encoder, shuffle, num_games)
        return itr

    def get_val_iter(self, batch_size, shuffle=False, num_games=None):
        recs_p = list(self.processed_p.glob(('val-%s-*.rec') % self.encoder.name()))
        itr = SGFIter(recs_p, batch_size, self.encoder, shuffle, num_games)
        return itr

    def get_test_iter(self, batch_size, shuffle=False, num_games=None):
        recs_p = list(self.processed_p.glob(('test-%s-*.rec') % self.encoder.name()))
        itr = SGFIter(recs_p, batch_size, self.encoder, shuffle, num_games)
        return itr

    def _extract(self, rec_p):
        record = mx.recordio.MXRecordIO(str(rec_p), 'r')
        size = int(record.read())
        data = [None]*size
        label = [None]*size
        for i in range(size):
            d = record.read()
            hdr, s = mx.recordio.unpack(d)
            label[i] = int(hdr.label)
            data[i] = np.reshape(np.frombuffer(s, dtype='int'), self.encoder.shape())
        return data, label

    def _get_data(self, recs_p):
        data = []
        label = []
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        try:
            it = pool.imap(self._extract, recs_p)
            for f, l in it:
                data.extend(f)
                label.extend(l)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(1)
        data = mx.nd.array(data)
        label = mx.nd.array(label)
        return data, label

    def get_train_array(self, shuffle=False, num_games=None):
        recs_p = list(self.processed_p.glob(('train-%s-*.rec') % self.encoder.name()))
        if num_games is not None:
            recs_p = random.sample(recs_p, num_games)
        if shuffle:
            random.shuffle(recs_p)
        return self._get_data( recs_p)

    def get_val_array(self, shuffle=False, num_games=None):
        recs_p = list(self.processed_p.glob(('val-%s-*.rec') % self.encoder.name()))
        if num_games is not None:
            recs_p = random.sample(recs_p, num_games)
        if shuffle:
            random.shuffle(recs_p)
        return self._get_data( recs_p)

    def get_test_array(self, shuffle=False, num_games=None):
        recs_p = list(self.processed_p.glob(('test-%s-*.rec') % self.encoder.name()))
        if num_games is not None:
            recs_p = random.sample(recs_p, num_games)
        if shuffle:
            random.shuffle(recs_p)
        return self._get_data( recs_p)


def _worker_fn(rec_p, encoder):
    record = mx.recordio.MXRecordIO(str(rec_p), 'r')
    size = int(record.read())
    data = [None]*size
    label = [None]*size
    for i in range(size):
        d = record.read()
        hdr, s = mx.recordio.unpack(d)
        label[i] = int(hdr.label)
        data[i] = np.reshape(np.frombuffer(s, dtype='int'), encoder.shape())
    record.close()
    return data, label


class SGFIter(mx.io.DataIter):
    def __init__(self, recs_p, batch_size, encoder, shuffle, num_games):
        super(SGFIter, self).__init__()
        self._batch_size = batch_size
        self._base_recs_p = recs_p
        self._shuffle = shuffle
        self._num_games = num_games
        self._encoder = encoder
        self._sent_idx = 0
        self._rcv_idx = 0
        self._buffer = {}
        self._data = []
        self._label = []
        # a games contains 100 moves at average
        self._recs_per_batch = ceil(self._batch_size / 100)
        # prefetch at least 10 batches
        self._prefetch = 10 * self._recs_per_batch
        cores = ceil(multiprocessing.cpu_count()/2)
        self._pool = multiprocessing.Pool(processes=cores)
        self._provide_data = [mx.io.DataDesc('data', (self._batch_size,) + self._encoder.shape(), dtype='int', layout='NCHW')]
        self._provide_label = [mx.io.DataDesc('label', (self._batch_size,), dtype='int', layout='NCHW')]
        self._recs_p = self._base_recs_p if self._num_games is None else random.sample(self._base_recs_p, self._num_games)
        self._max_recs_idx = len(self._recs_p)
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            for _ in range(self._prefetch):
                self._push_next()
            while self._rcv_idx in self._buffer:
                self._fetch_next()
            if self._rcv_idx == self._sent_idx and self._sent_idx == self._max_recs_idx and not self._buffer and 0 == len(self._label):
                assert not self._buffer, "data buffer should be empty at this moment"
                # reached end of data/label lists
                raise StopIteration
            # get batch-sized lists of data and label
            if (0 == len(self._label)):
                print('continue')
                continue
            data = self._data[:self._batch_size]
            label = self._label[:self._batch_size]
            self._data = self._data[self._batch_size:]
            self._label = self._label[self._batch_size:]
            # increment iteration index
            pad = 0
            if len(label) < self._batch_size:
                pad = self._batch_size - len(label)
                data = data + [data[0]]*pad
                label = label + [label[0]]*pad
            data = [mx.nd.array(data)]
            label = [mx.nd.array(label)]
            return mx.io.DataBatch(data, label, pad=pad)

    def _fetch_next(self):
        if self._rcv_idx >= self._sent_idx:
            return
        assert self._rcv_idx in self._buffer, ''
        ret = self._buffer.pop(self._rcv_idx)
        try:
            data, label = ret.get()
            self._data.extend(data)
            self._label.extend(label)
            self._rcv_idx += 1
        except Exception:
            self._pool.terminate()
            raise

    def _push_next(self):
        if self._sent_idx == self._max_recs_idx:
            return
        rec_p = self._recs_p[self._sent_idx]
        async_ret = self._pool.apply_async(_worker_fn, (rec_p, self._encoder))
        self._buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def reset(self):
        self._sent_idx = 0
        self._rcv_idx = 0
        self._buffer = {}
        self._data = []
        self._label = []
        if self._shuffle:
            random.shuffle(self._recs_p)
        for _ in range(self._prefetch):
            self._push_next()
        while self._rcv_idx in self._buffer:
            self._fetch_next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
