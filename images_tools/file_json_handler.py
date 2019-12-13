import json

from images_tools import file_base


class JsonHandler(file_base.BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, **kwargs)
