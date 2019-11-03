import os
import shutil

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from kigo.data.index_processor import KGSIndex


def download(data_p):
    shutil.rmtree(data_p, ignore_errors=True)
    data_p.mkdir()
    index = KGSIndex(data_p)
    index.download_files()


def main():
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    data_p = Path(os.environ.get('PATH_RAW')).resolve()
    download(data_p)


if __name__ == '__main__':
    main()
