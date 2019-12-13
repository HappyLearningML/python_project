#-*-coding:utf-8-*-
import os

def download(path):
    """Use urllib to download a file.

    Parameters
    ----------
    path : str
        Url to download

    Returns
    -------
    path : str
        Location of downloaded file.
    """
    from six.moves import urllib

    fname = path.split('/')[-1]
    if os.path.exists(fname):
        return fname

    print('Downloading ' + path)

    def progress(count, block_size, total_size):
        if count % 20 == 0:
            print('Downloaded %02.02f/%02.02f MB' % (
                count * block_size / 1024.0 / 1024.0,
                total_size / 1024.0 / 1024.0), end='\r')

    filepath, _ = urllib.request.urlretrieve(
        path, filename=fname, reporthook=progress)
    return filepath


def download_and_extract_tar(path, dst):
    """Download and extract a tar file.

    Parameters
    ----------
    path : str
        Url to tar file to download.
    dst : str
        Location to save tar file contents.
    """
    import tarfile
    filepath = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
        tarfile.open(filepath, 'r:gz').extractall(dst)


def download_and_extract_zip(path, dst):
    """Download and extract a zip file.

    Parameters
    ----------
    path : str
        Url to zip file to download.
    dst : str
        Location to save zip file contents.
    """
    import zipfile
    filepath = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
        zf = zipfile.ZipFile(file=filepath)
        zf.extractall(dst)