'''
                    Copyright Oliver Kowalke 2019.
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
import numpy as np
import os
import shutil
import tarfile

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from random import shuffle

from kigo.encoders.oneplane import OnePlaneEncoder
from kigo.data.records_creator import SGFTFRecordCreator


def extract_sgfs(dir_p):
    zips_p = dir_p.glob('*.tar.gz')
    sgfs = []
    for zip_p in zips_p:
        with tarfile.open(zip_p) as tar:
            names = tar.getnames()
            for name in names:
                if not name.endswith('.sgf'):
                    continue
                sgfs.append(tar.extractfile(name).read())
    return sgfs


def split_data(sgfs, train_frac, val_frac):
    # create array of indices
    # each index represents one spreadsheet (e.g. jpg)
    indices = np.arange(0, len(sgfs))
    # split indices in training/validation/testing subsets
    train_indices, test_indices, val_indices = np.split(indices, [int(train_frac * len(indices)), int((1 - val_frac) * len(indices))])
    # split sgfs according to the indices
    train_sgfs = sgfs[train_indices[0]:train_indices[-1]+1]
    test_sgfs = sgfs[test_indices[0]:test_indices[-1]+1]
    val_sgfs = sgfs[val_indices[0]:val_indices[-1]+1]
    return train_sgfs, val_sgfs, test_sgfs


def main():
    # environment
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    # parameters
    raw_p = Path(os.environ.get('PATH_RAW')).resolve()
    processed_p = Path(os.environ.get('PATH_PROCESSED')).resolve()
    train_frac = float(os.environ.get('TRAIN_FRAC'))
    val_frac = float(os.environ.get('VAL_FRAC'))
    sample_size = int(os.environ.get('SAMPLE_SIZE'))
    # logging
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    # get SGF data
    sgfs = extract_sgfs(raw_p)
    shuffle(sgfs)
    # split data into training, validation and test
    train_sgfs, val_sgfs, test_sgfs = split_data(sgfs, train_frac, val_frac)
    # encoder
    go_board_rows, go_board_cols = 19, 19
    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    # create training records
    tfr_creator = SGFTFRecordCreator(encoder, sample_size)
    tfr_creator.create(processed_p.joinpath('train'), train_sgfs)
    tfr_creator.create(processed_p.joinpath('val'), val_sgfs)
    tfr_creator.create(processed_p.joinpath('test'), test_sgfs)


if __name__ == '__main__':
    main()
