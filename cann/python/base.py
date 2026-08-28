import os
import sys
from abc import ABC, abstractmethod

path = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.join(path, "..")
sys.path.insert(0, os.path.join(_parent, "examples/samples/python/common/"))
sys.path.insert(0, os.path.join(_parent, "examples/samples/python/common/acllite"))

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
import acllite_utils as utils

_acl_resource = None


def _get_acl_resource():
    global _acl_resource
    if _acl_resource is None:
        _acl_resource = AclLiteResource()
        _acl_resource.init()
    return _acl_resource


def _release_acl_resource():
    global _acl_resource
    if _acl_resource is not None:
        try:
            if hasattr(_acl_resource, 'release'):
                _acl_resource.release()
            elif hasattr(_acl_resource, 'destroy'):
                _acl_resource.destroy()
        except Exception:
            pass
        _acl_resource = None


class BaseModel(ABC):
    def __init__(self, model_path):
        self._model_path = model_path
        self._model = None
        self._model_width = 0
        self._model_height = 0

    def init(self):
        _get_acl_resource()
        self._model = AclLiteModel(self._model_path)
        return 0

    def release(self):
        if self._model is not None:
            try:
                self._model.__del__()
            except Exception:
                pass
            self._model = None

    @abstractmethod
    def pre_process(self, image_path):
        pass

    @abstractmethod
    def inference(self, input_data):
        pass

    @abstractmethod
    def post_process(self, infer_output):
        pass

    def run(self, image_path):
        input_data = self.pre_process(image_path)
        output = self.inference(input_data)
        results = self.post_process(output)
        return results
