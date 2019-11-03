import multiprocessing
import os
import six
import sys

from pathlib import Path
from urllib.request import urlopen, urlretrieve


# parallelize data download via multiprocessing
def worker(url_and_target):
    try:
        (url, target_path) = url_and_target
        print('>>> Downloading ' + target_path)
        urlretrieve(url, target_path)
    except (KeyboardInterrupt, SystemExit):
        print('>>> Exiting child process')


# create an index of zip files containing SGF data of actual Go Games on KGS
class KGSIndex:
    def __init__(self,
                 data_p,
                 kgs_url='http://u-go.net/gamerecords/',
                 index_page='kgs_index.html'):
        # data_p: name of directory relative to current path to store SGF data
        # kgs_url: URL with links to zip files of games
        # index_page: Name of local html file of kgs_url
        self.data_p = data_p
        self.kgs_url = kgs_url
        self.index_page = data_p.joinpath(index_page)
        self.file_info = []
        self.urls = []

    def _load_index(self):
        print('load index')
        index_contents = self._create_index_page()
        split_page = [item for item in index_contents.split('<a href="') if item.startswith("https://")]
        for item in split_page:
            download_url = item.split('">Download')[0]
            if download_url.endswith('.tar.gz'):
                self.urls.append(download_url)
        for url in self.urls:
            filename = os.path.basename(url)
            split_file_name = filename.split('-')
            num_games = int(split_file_name[len(split_file_name) - 2])
            print(filename + ' ' + str(num_games))
            self.file_info.append({'url': url, 'filename': filename, 'num_games': num_games})

    def _create_index_page(self):
        if self.index_page.exists():
            print('>>> Reading cached index page')
            with open(self.index_page, 'r') as f:
                index_contents = f.read()
        else:
            print('>>> Downloading index page')
            with urlopen(self.kgs_url) as fp:
                index_contents = six.text_type(fp.read())
            with open(self.index_page, 'w') as f:
                f.write(index_contents)
        return index_contents

    def download_files(self):
        assert self.data_p.exists()
        assert self.data_p.is_dir()
        # load index on creation
        self._load_index()
        # fetch data
        urls_to_download = []
        for file_info in self.file_info:
            url = file_info['url']
            file_name = self.data_p.joinpath(file_info['filename']);
            if not file_name.exists():
                urls_to_download.append((url, str(file_name)))
        size = len(urls_to_download)
        if 0 < size:
            cores = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=size)
            try:
                it = pool.imap(worker, urls_to_download)
                for _ in it:
                    pass
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print(">>> Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
                pool.join()
                sys.exit(1)
