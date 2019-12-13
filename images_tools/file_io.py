from images_tools import file_base, file_json_handler, file_pickle_handler, file_yaml_handler
from sys_tools import misc

file_handlers = {
    'json': file_json_handler.JsonHandler(),
    'yaml': file_yaml_handler.YamlHandler(),
    'yml': file_yaml_handler.YamlHandler(),
    'pickle': file_pickle_handler.PickleHandler(),
    'pkl': file_pickle_handler.PickleHandler()
}


def load(file, file_format=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or file-like object): Filename or a file-like object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".

    Returns:
        The content from the file.
    """
    if file_format is None and misc.is_str(file):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    if misc.is_str(file):
        obj = handler.load_from_path(file, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Args:
        obj (any): The python object to be dumped.
        file (str or file-like object, optional): If not specified, then the
            object is dump to a str, otherwise to a file specified by the
            filename or file-like object.
        file_format (str, optional): Same as :func:`load`.

    Returns:
        bool: True for success, False otherwise.
    """
    if file_format is None:
        if misc.is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif misc.is_str(file):
        handler.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def _register_handler(handler, file_formats):
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, file_base.BaseFileHandler):
        raise TypeError(
            'handler must be a child of BaseFileHandler, not {}'.format(
                type(handler)))
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not misc.is_list_of(file_formats, str):
        raise TypeError('file_formats must be a str or a list of str')
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats, **kwargs):

    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
