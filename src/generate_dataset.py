import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from kigo.data.dataset_builder import SGFDataset
from kigo.encoders.sevenplane import SevenPlaneEncoder


def main():
    # env
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    train_frac = float(os.environ.get('TRAIN_FRAC'))
    val_frac = float(os.environ.get('VAL_FRAC'))

    sgf_files = 0
    go_board_rows, go_board_cols = 19, 19
    prefetch_size = tf.data.experimental.AUTOTUNE

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))

    builder = SGFDataset(encoder, train_frac, val_frac, sgf_files)
    dl_config = tfds.download.DownloadConfig(register_checksums=True)
    builder.download_and_prepare(download_config=dl_config)

if __name__ == '__main__':
    main()
