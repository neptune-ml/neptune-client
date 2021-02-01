#
# Copyright (c) 2020, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import atexit
import threading
import time
import traceback
import uuid
from contextlib import AbstractContextManager
from datetime import datetime
from io import IOBase
from typing import Dict, Any, Union, List, Optional

from neptune.alpha.attributes.atoms.datetime import Datetime as DatetimeAttr
from neptune.alpha.attributes.atoms.experiment_state import ExperimentState as ExperimentStateAttr
from neptune.alpha.attributes.atoms.file import File as FileAttr
from neptune.alpha.attributes.atoms.float import Float as FloatAttr
from neptune.alpha.attributes.atoms.git_ref import GitRef as GitRefAttr
from neptune.alpha.attributes.atoms.string import String as StringAttr
from neptune.alpha.attributes.attribute import Attribute
from neptune.alpha.attributes.file_set import FileSet as FileSetAttr
from neptune.alpha.attributes.series.float_series import FloatSeries as FloatSeriesAttr
from neptune.alpha.attributes.series.image_series import ImageSeries as ImageSeriesAttr
from neptune.alpha.attributes.series.string_series import StringSeries as StringSeriesAttr
from neptune.alpha.attributes.sets.string_set import StringSet as StringSetAttr
from neptune.alpha.exceptions import MetadataInconsistency
from neptune.alpha.handler import Handler
from neptune.alpha.internal.backends.api_model import AttributeType
from neptune.alpha.internal.backends.neptune_backend import NeptuneBackend
from neptune.alpha.internal.background_job import BackgroundJob
from neptune.alpha.internal.experiment_structure import ExperimentStructure
from neptune.alpha.internal.operation import DeleteAttribute
from neptune.alpha.internal.operation_processors.operation_processor import OperationProcessor
from neptune.alpha.internal.utils import verify_type, get_stream_content,\
    is_stream, is_float, is_string, is_float_like, is_string_like
from neptune.alpha.internal.utils.paths import parse_path
from neptune.alpha.internal.value_to_attribute_visitor import ValueToAttributeVisitor
from neptune.alpha.types.atoms.datetime import Datetime
from neptune.alpha.types.atoms.float import Float
from neptune.alpha.types.atoms.string import String
from neptune.alpha.types.atoms.file import File
from neptune.alpha.types.value import Value


class Experiment(AbstractContextManager):

    def __init__(
            self,
            _uuid: uuid.UUID,
            backend: NeptuneBackend,
            op_processor: OperationProcessor,
            background_job: BackgroundJob
    ):
        self._uuid = _uuid
        self._backend = backend
        self._op_processor = op_processor
        self._bg_job = background_job
        self._structure = ExperimentStructure[Attribute]()
        self._lock = threading.RLock()
        self._started = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        traceback.print_exception(exc_type, exc_val, exc_tb)
        self.stop()

    def __getitem__(self, path: str) -> 'Handler':
        return Handler(self, path)

    def __setitem__(self, key: str, value) -> None:
        self.__getitem__(key).assign(value)

    def __delitem__(self, path) -> None:
        self.pop(path)

    def ping(self):
        self._backend.ping_experiment(self._uuid)

    def start(self):
        atexit.register(self._shutdown_hook)
        self._op_processor.start()
        self._bg_job.start(self)
        self._started = True

    def stop(self, seconds: Optional[Union[float, int]] = None):
        verify_type("seconds", seconds, (float, int, type(None)))
        if not self._started:
            return
        self._started = False
        ts = time.time()
        self._bg_job.stop()
        self._bg_job.join(seconds)
        with self._lock:
            sec_left = None if seconds is None else seconds - (time.time() - ts)
            self._op_processor.stop(sec_left)

    def get_structure(self) -> Dict[str, Any]:
        return self._structure.get_structure()

    def define(self,
               path: str,
               value: Union[Value, int, float, str, datetime, IOBase],
               wait: bool = False
               ) -> Attribute:
        if isinstance(value, Value):
            pass
        elif is_float(value):
            value = Float(value)
        elif is_string(value):
            value = String(value)
        elif isinstance(value, datetime):
            value = Datetime(value)
        elif is_stream(value):
            file_content, file_name = get_stream_content(value)
            value = File(file_content=file_content, file_name=file_name)
        elif is_float_like(value):
            value = Float(float(value))
        elif is_string_like(value):
            value = String(str(value))
        else:
            raise TypeError("Value of unsupported type {}".format(type(value)))
        parsed_path = parse_path(path)

        with self._lock:
            old_attr = self._structure.get(parsed_path)
            if old_attr:
                raise MetadataInconsistency("Attribute {} is already defined".format(path))
            attr = ValueToAttributeVisitor(self, parsed_path).visit(value)
            attr.assign(value, wait)
            self._structure.set(parsed_path, attr)
            return attr

    def get_attribute(self, path: str) -> Optional[Attribute]:
        with self._lock:
            return self._structure.get(parse_path(path))

    def set_attribute(self, path: str, attribute: Attribute) -> Optional[Attribute]:
        with self._lock:
            return self._structure.set(parse_path(path), attribute)

    def exists(self, path: str) -> bool:
        verify_type("path", path, str)
        return self.get_attribute(path) is not None

    def pop(self, path: str, wait: bool = False):
        verify_type("path", path, str)
        with self._lock:
            parsed_path = parse_path(path)
            self._op_processor.enqueue_operation(DeleteAttribute(parsed_path), wait)
            self._structure.pop(parsed_path)

    def lock(self) -> threading.RLock:
        return self._lock

    def wait(self, disk_only=False):
        with self._lock:
            if disk_only:
                self._op_processor.flush()
            else:
                self._op_processor.wait()

    def sync(self, wait: bool = True):
        with self._lock:
            if wait:
                self._op_processor.wait()
            attributes = self._backend.get_attributes(self._uuid)
            self._structure.clear()
            for attribute in attributes:
                self._define_attribute(parse_path(attribute.path), attribute.type)

    def _define_attribute(self, _path: List[str], _type: AttributeType):
        if _type == AttributeType.FLOAT:
            self._structure.set(_path, FloatAttr(self, _path))
        if _type == AttributeType.STRING:
            self._structure.set(_path, StringAttr(self, _path))
        if _type == AttributeType.DATETIME:
            self._structure.set(_path, DatetimeAttr(self, _path))
        if _type == AttributeType.FILE:
            self._structure.set(_path, FileAttr(self, _path))
        if _type == AttributeType.FILE_SET:
            self._structure.set(_path, FileSetAttr(self, _path))
        if _type == AttributeType.FLOAT_SERIES:
            self._structure.set(_path, FloatSeriesAttr(self, _path))
        if _type == AttributeType.STRING_SERIES:
            self._structure.set(_path, StringSeriesAttr(self, _path))
        if _type == AttributeType.IMAGE_SERIES:
            self._structure.set(_path, ImageSeriesAttr(self, _path))
        if _type == AttributeType.STRING_SET:
            self._structure.set(_path, StringSetAttr(self, _path))
        if _type == AttributeType.GIT_REF:
            self._structure.set(_path, GitRefAttr(self, _path))
        if _type == AttributeType.EXPERIMENT_STATE:
            self._structure.set(_path, ExperimentStateAttr(self, _path))

    def _shutdown_hook(self):
        self.stop()
