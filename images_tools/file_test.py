import os
import os.path as osp
import tempfile

import pytest
from images_tools import file_base, file_json_handler, file_pickle_handler, file_yaml_handler, file_io, file_parse


def _test_handler(file_format, test_obj, str_checker, mode='r+'):
    # dump to a string
    dump_str = file_io.dump(test_obj, file_format=file_format)
    str_checker(dump_str)

    # load/dump with filenames
    tmp_filename = osp.join(tempfile.gettempdir(), 'mmcv_test_dump')
    file_io.dump(test_obj, tmp_filename, file_format=file_format)
    assert osp.isfile(tmp_filename)
    load_obj = file_io.load(tmp_filename, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # json load/dump with a file-like object
    with tempfile.NamedTemporaryFile(mode, delete=False) as f:
        tmp_filename = f.name
        file_io.dump(test_obj, f, file_format=file_format)
    assert osp.isfile(tmp_filename)
    with open(tmp_filename, mode) as f:
        load_obj = file_io.load(f, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # automatically inference the file format from the given filename
    tmp_filename = osp.join(tempfile.gettempdir(),
                            'mmcv_test_dump.' + file_format)
    file_io.dump(test_obj, tmp_filename)
    assert osp.isfile(tmp_filename)
    load_obj = file_io.load(tmp_filename)
    assert load_obj == test_obj
    os.remove(tmp_filename)


obj_for_test = [{'a': 'abc', 'b': 1}, 2, 'c']


def test_json():

    def json_checker(dump_str):
        assert dump_str in [
            '[{"a": "abc", "b": 1}, 2, "c"]', '[{"b": 1, "a": "abc"}, 2, "c"]'
        ]

    _test_handler('json', obj_for_test, json_checker)


def test_yaml():

    def yaml_checker(dump_str):
        assert dump_str in [
            '- {a: abc, b: 1}\n- 2\n- c\n', '- {b: 1, a: abc}\n- 2\n- c\n',
            '- a: abc\n  b: 1\n- 2\n- c\n', '- b: 1\n  a: abc\n- 2\n- c\n'
        ]

    _test_handler('yaml', obj_for_test, yaml_checker)


def test_pickle():

    def pickle_checker(dump_str):
        import pickle
        assert pickle.loads(dump_str) == obj_for_test

    _test_handler('pickle', obj_for_test, pickle_checker, mode='rb+')


def test_exception():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']

    with pytest.raises(ValueError):
        file_io.dump(test_obj)

    with pytest.raises(TypeError):
        file_io.dump(test_obj, 'tmp.txt')


def test_register_handler():

    @file_io.register_handler('txt')
    class TxtHandler1(file_base.BaseFileHandler):

        def load_from_fileobj(self, file):
            return file.read()

        def dump_to_fileobj(self, obj, file):
            file.write(str(obj))

        def dump_to_str(self, obj, **kwargs):
            return str(obj)

    @file_io.register_handler(['txt1', 'txt2'])
    class TxtHandler2(file_base.BaseFileHandler):

        def load_from_fileobj(self, file):
            return file.read()

        def dump_to_fileobj(self, obj, file):
            file.write('\n')
            file.write(str(obj))

        def dump_to_str(self, obj, **kwargs):
            return str(obj)

    content = file_io.load(osp.join(osp.dirname(__file__), 'data/filelist.txt'))
    assert content == '1.jpg\n2.jpg\n3.jpg\n4.jpg\n5.jpg'
    tmp_filename = osp.join(tempfile.gettempdir(), 'mmcv_test.txt2')
    file_io.dump(content, tmp_filename)
    with open(tmp_filename, 'r') as f:
        written = f.read()
    os.remove(tmp_filename)
    assert written == '\n' + content


def test_list_from_file():
    filename = osp.join(osp.dirname(__file__), 'data/filelist.txt')
    filelist = file_parse.list_from_file(filename)
    assert filelist == ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    filelist = file_parse.list_from_file(filename, prefix='a/')
    assert filelist == ['a/1.jpg', 'a/2.jpg', 'a/3.jpg', 'a/4.jpg', 'a/5.jpg']
    filelist = file_parse.list_from_file(filename, offset=2)
    assert filelist == ['3.jpg', '4.jpg', '5.jpg']
    filelist = file_parse.list_from_file(filename, max_num=2)
    assert filelist == ['1.jpg', '2.jpg']
    filelist = file_parse.list_from_file(filename, offset=3, max_num=3)
    assert filelist == ['4.jpg', '5.jpg']


def test_dict_from_file():
    filename = osp.join(osp.dirname(__file__), 'data/mapping.txt')
    mapping = file_parse.dict_from_file(filename)
    assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
    mapping = file_parse.dict_from_file(filename, key_type=int)
    assert mapping == {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
