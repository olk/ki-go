import shutil

from pathlib import Path
from kigo.data.index_processor import KGSIndex


def download(data_p):
    shutil.rmtree(data_p, ignore_errors=True)
    data_p.mkdir()
    index = KGSIndex(data_p)
    index.download_files()


def main():
    data_p = Path('raw').resolve()
    download(data_p)


if __name__ == '__main__':
    main()
