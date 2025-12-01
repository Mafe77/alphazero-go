from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import multiprocessing
import six
if sys.version_info[0] == 3:
    from urllib.request import urlopen, urlretrieve
else:
    from urllib import urlopen, urlretrieve


def worker(url_and_target):  # Parallelize data download via multiprocessing
    try:
        (url, target_path) = url_and_target
        print('>>> Downloading ' + target_path)
        urlretrieve(url, target_path)
    except (KeyboardInterrupt, SystemExit):
        print('>>> Exiting child process')


class KGSIndex:

    def __init__(self,
                 kgs_url='http://u-go.net/gamerecords/',
                 index_page='kgs_index.html',
                 data_directory='data'):
        """Create an index of zip files containing SGF data of actual Go Games on KGS.

        Parameters:
        -----------
        kgs_url: URL with links to zip files of games
        index_page: Name of local html file of kgs_url
        data_directory: name of directory relative to current path to store SGF data
        """
        self.kgs_url = kgs_url
        self.index_page = index_page
        self.data_directory = data_directory
        self.file_info = []
        self.urls = []
        self.load_index()  # Load index on creation

    def download_files(self):
        """Download zip files by distributing work across all CPU cores."""
        os.makedirs(self.data_directory, exist_ok=True)

        # Build list of (url, path)
        urls_to_download = []
        for file_info in self.file_info:
            url = file_info['url']
            file_name = file_info['filename']
            target = os.path.join(self.data_directory, file_name)

            if not os.path.isfile(target):
                urls_to_download.append((url, target))

        if not urls_to_download:
            print(">>> All files already downloaded.")
            return

        cores = multiprocessing.cpu_count()
        print(f">>> Starting download with {cores} parallel workers...")

        pool = multiprocessing.Pool(processes=cores)

        try:
            # imap_unordered gives results as soon as any worker finishes
            for _ in pool.imap_unordered(worker, urls_to_download):
                pass

        except KeyboardInterrupt:
            print(">>> KeyboardInterrupt detected! Terminating workers...")
            pool.terminate()

        except Exception as e:
            print(">>> Error occurred:", e)
            pool.terminate()

        finally:
            pool.close()
            pool.join()
            print(">>> All downloads complete.")


    def create_index_page(self):
        """If there is no local html containing links to files, create one."""
        if os.path.isfile(self.index_page):
            print('>>> Reading cached index page')
            index_file = open(self.index_page, 'r')
            index_contents = index_file.read()
            index_file.close()
        else:
            print('>>> Downloading index page')
            fp = urlopen(self.kgs_url)
            data = fp.read().decode('utf-8')   # <--- FIX
            fp.close()
            index_contents = data
            with open(self.index_page, 'w', encoding='utf-8') as index_file:
                index_file.write(index_contents)
            index_file.close()
        return index_contents

    def load_index(self):
        """Create the actual index representation from the previously downloaded or cached html."""
        index_contents = self.create_index_page()
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


if __name__ == '__main__':
    index = KGSIndex()
    index.download_files()