import os
import urllib.request
import zipfile
from pathlib import Path

import progressbar

from ..constants import DATASET_URLS, PLANES

_progress_bar = None


def _show_progress(block_num, block_size, total_size):
    global _progress_bar
    if _progress_bar is None:
        _progress_bar = progressbar.ProgressBar(maxval=total_size)
        _progress_bar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        _progress_bar.update(downloaded)
    else:
        _progress_bar.finish()
        _progress_bar = None


def download_file(url: str, disk_path: Path) -> None:
    print(f"downloading {url}")
    filename, _ = urllib.request.urlretrieve(url, reporthook=_show_progress)
    os.makedirs(disk_path, exist_ok=True)
    with zipfile.ZipFile(filename, "r") as zf:
        zf.extractall(disk_path)


def download_data(to: Path = Path("dataset"), planes=PLANES, splits=("train", "test")) -> None:
    for plane in planes:
        for split in splits:
            download_file(DATASET_URLS[plane][split], Path(to) / plane / split)
