from pathlib import Path

from kigo.data.dataset import SGFDatasetBuilder
from kigo.encoders.sevenplane import SevenPlaneEncoder


def main():
    data_p = Path('/storage/data/ki-go').resolve()
    encoder = SevenPlaneEncoder((19, 19))
    builder = SGFDatasetBuilder(data_p, encoder)
    builder.download_and_prepare()

if __name__ == '__main__':
    main()
