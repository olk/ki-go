import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from pathlib import Path

from kigo.data.tf.dataset_builder import SGFDatasetBuilder
from kigo.encoders.sevenplane import SevenPlaneEncoder


def main():
    go_board_rows, go_board_cols = 19, 19
    prefetch_size = tf.data.experimental.AUTOTUNE

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))

    builder = SGFDatasetBuilder(encoder, sgf_files=10)
    dl_config = tfds.download.DownloadConfig(register_checksums=True)
    builder.download_and_prepare(download_config=dl_config)

if __name__ == '__main__':
    main()
