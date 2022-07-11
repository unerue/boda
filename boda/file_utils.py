import json
import os
import sys
from urllib.request import urlretrieve


class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, list):
            return json.JSONEncoder().encode(obj)

        return json.JSONEncoder.default(self, obj)


def progressbar(cur, total=100):
    percent = "{:.2%}".format(cur / total)
    sys.stdout.write("\r")
    # sys.stdout.write("[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)),percent))
    sys.stdout.write("[%-100s] %s" % ("=" * int(cur), percent))
    sys.stdout.flush()


def schedule(blocknum, blocksize, totalsize):
    """
    blocknum: currently downloaded block
         blocksize: block size for each transfer
         totalsize: total size of web page files
    """
    if totalsize == 0:
        percent = 0
    else:
        percent = blocknum * blocksize / totalsize
    if percent > 1.0:
        percent = 1.0

    percent = percent * 100
    print("download : %.2f%%" % (percent))
    progressbar(percent)


def reporthook(count, block_size, total_size):
    """
    https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    """
    # global start_time
    # if count == 0:
    #     start_time = time.time()
    #     return
    # duration = time.time() - start_time
    progress_size = int(count * block_size)
    # speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    # min(int(count*blockSize*100/totalSize),100)
    sys.stdout.write(
        f"\rDownload file for pretrained model: {percent:>3}% {progress_size / (1024*1204):>4.1f} MB"
    )

    # sys.stdout.write("\rDownload pretrained model: %d%%, %d MB, %d KB/s, %d seconds passed" %
    #                 (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def get_file_from_url(
    file_name: str,
):
    """
    file_name (): model_name/file_name.json or pth
    """
    url = "https://unerue.synology.me/boda/models/"
    urlretrieve(f"{url}{file_name}", config_file, reporthook)
    print()
