import os
import os.path as osp
import sys
from pathlib import Path

import six
import pytest

def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

if sys.version_info <= (3, 3):
    FileNotFoundError = IOError
else:
    FileNotFoundError = FileNotFoundError


def is_filepath(x):
    if is_str(x) or isinstance(x, Path):
        return True
    else:
        return False


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not osp.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def _scandir_py35(dir_path, suffix=None):
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        filename = entry.name
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def _scandir_py(dir_path, suffix=None):
    for filename in os.listdir(dir_path):
        if not osp.isfile(osp.join(dir_path, filename)):
            continue
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def scandir(dir_path, suffix=None):
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    if sys.version_info >= (3, 5):
        return _scandir_py35(dir_path, suffix)
    else:
        return _scandir_py(dir_path, suffix)



def test_is_filepath():
    assert is_filepath(__file__)
    assert is_filepath('abc')
    assert is_filepath(Path('/etc'))
    assert not is_filepath(0)


def test_fopen():
    assert hasattr(fopen(__file__), 'read')
    assert hasattr(fopen(Path(__file__)), 'read')


def test_check_file_exist():
    check_file_exist(__file__)
    if sys.version_info > (3, 3):
        with pytest.raises(FileNotFoundError):  # noqa
            check_file_exist('no_such_file.txt')
    else:
        with pytest.raises(IOError):
            check_file_exist('no_such_file.txt')


def test_scandir():
    folder = osp.join(osp.dirname(__file__), 'data/for_scan')
    filenames = ['a.bin', '1.txt', '2.txt', '1.json', '2.json']

    assert set(scandir(folder)) == set(filenames)
    assert set(scandir(folder, '.txt')) == set(
        [filename for filename in filenames if filename.endswith('.txt')])
    assert set(scandir(folder, ('.json', '.txt'))) == set([
        filename for filename in filenames
        if filename.endswith(('.txt', '.json'))
    ])
    assert set(scandir(folder, '.png')) == set()
    with pytest.raises(TypeError):
        scandir(folder, 111)
