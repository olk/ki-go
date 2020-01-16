import mxnet as mx
import numpy as np
import os
import random
import shutil
import six
import tarfile
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
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
        # create recordio file
        size = len(data)
        assert len(label) == size
        rec_p = self.processed_p.joinpath('%s-%s-%d' % (self.encoder.name(), sgf_p.stem, size)).with_suffix('.rec')
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
        print('encoding %d SGF files' % len(sgfs_p))
        with ProcessPoolExecutor() as pool:
            pool.map(self._encode_and_persist, sgfs_p, chunksize=100)
        # split and rename
        recs_p = list(self.processed_p.glob(self.encoder.name() + '-*.rec'))
        print('created %d record files' % len(recs_p))
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

    def as_train_dataset(self, batch_size, num_segments=1, shuffle=False):
        recs_p = list(self.processed_p.glob(('train-%s-*.rec') % self.encoder.name()))
        if shuffle:
            random.shuffle(recs_p)
        size = len(recs_p)
        steps = int(size/num_segments)
        segments = [recs_p[i*steps:(i+1)*steps] for i in range(steps+1)]
        ds = SGFDataSet(segments, batch_size, self.encoder)
        return ds

    def as_val_dataset(self, batch_size, num_segments=1, shuffle=False):
        recs_p = list(self.processed_p.glob(('val-%s-*.rec') % self.encoder.name()))
        if shuffle:
            random.shuffle(recs_p)
        size = len(recs_p)
        steps = int(size/num_segments)
        segments = [recs_p[i*steps:(i+1)*steps] for i in range(steps+1)]
        ds = SGFDataSet(segments, batch_size, self.encoder)
        return ds

    def as_test_dataset(self, batch_size, num_segments=1, shuffle=False):
        recs_p = list(self.processed_p.glob(('test-%s-*.rec') % self.encoder.name()))
        if shuffle:
            random.shuffle(recs_p)
        size = len(recs_p)
        steps = int(size/num_segments)
        segments = [recs_p[i*steps:(i+1)*steps] for i in range(steps+1)]
        ds = SGFDataSet(segments, batch_size, self.encoder)
        return ds


_shape = None
_decoder = None
_dataset = None

def _init_decoder(dataset, shape):
    global _dataset
    global _shape
    _dataset = dataset
    _shape = shape

def _decode_fn(sidx, ridx):
    global _dataset
    global _shape
    assert _dataset is not None
    assert _shape is not None
    record = mx.recordio.MXRecordIO(str(_dataset[sidx][ridx]), 'r')
    size = int(record.read())
    data = [None]*size
    label = [None]*size
    for i in range(size):
        d = record.read()
        hdr, s = mx.recordio.unpack(d)
        label[i] = int(hdr.label)
        data[i] = np.reshape(np.frombuffer(s, dtype='int'), _shape)
    record.close()
    return data, label

def _init_collector(dataset, shape):
    global _dataset
    global _decoder
    _dataset = dataset
    _decoder = ProcessPoolExecutor(initializer=_init_decoder, initargs=(dataset, shape))

def _fetch_fn(sidx):
    global _dataset
    global _decoder
    assert _dataset is not None
    assert _decoder is not None
    # push data
    results = [_decoder.submit(_decode_fn, sidx, ridx) for ridx in range(len(_dataset[sidx]))]
    # fetch data
    data = []
    label = []
    assert len(results) == len(_dataset[sidx]), 'results and current segment of different length'
    for f in as_completed(results):
        d, l = f.result()
        del f
        data.extend(d)
        label.extend(l)
    assert len(data) == len(label), 'data and label of different length'
    return data, label

class SGFDataSet:
    def __init__(self, dataset, batch_size, encoder):
        self._batch_size = batch_size
        self._max_idx = len(dataset)
        self._idx = 0
        self._collector = ProcessPoolExecutor(max_workers=1, initializer=_init_collector, initargs=(dataset, encoder.shape(),))

    def __iter__(self):
        self._fut = self._collector.submit(_fetch_fn, self._idx)
        return self

    def __next__(self):
        if self._idx == self._max_idx:
            raise StopIteration
        # processing of current segment has finished
        data, label = self._fut.result()
        data = mx.nd.array(data)
        label = mx.nd.array(label)
        itr = mx.io.NDArrayIter(data, label, self._batch_size)
        # next segment
        self._idx += 1
        # decode next segment
        self._fut = self._collector.submit(_fetch_fn, self._idx)
        return itr
