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
from datetime import datetime
from io import IOBase
from typing import TYPE_CHECKING, Union, Iterable

from neptune.alpha.attributes.file_set import FileSet
from neptune.alpha.attributes.series import ImageSeries
from neptune.alpha.attributes.series.float_series import FloatSeries
from neptune.alpha.attributes.series.string_series import StringSeries
from neptune.alpha.attributes.sets.string_set import StringSet
from neptune.alpha.internal.utils import verify_type, is_collection, verify_collection_type, is_float, is_string, \
    is_float_like, is_string_like
from neptune.alpha.internal.utils.paths import join_paths, parse_path
from neptune.alpha.types.atoms.file import File
from neptune.alpha.types.series.image import Image
from neptune.alpha.types.value import Value

if TYPE_CHECKING:
    from neptune.alpha.experiment import Experiment


class Handler:

    def __init__(self, _experiment: 'Experiment', path: str):
        super().__init__()
        self._experiment = _experiment
        self._path = path

    def __getitem__(self, path: str) -> 'Handler':
        return Handler(self._experiment, join_paths(self._path, path))

    def __setitem__(self, key: str, value) -> None:
        self[key].assign(value)

    def __getattr__(self, attribute_name):
        attr = self._experiment.get_attribute(self._path)
        if attr:
            return getattr(attr, attribute_name)
        else:
            raise AttributeError()

    def assign(self, value: Union[Value, int, float, str, datetime, IOBase, dict], wait: bool = False) -> None:
        with self._experiment.lock():
            attr = self._experiment.get_attribute(self._path)
            if attr:
                attr.assign(value, wait)
            else:
                self._experiment.define(self._path, value, wait)

    def save(self, value: str, wait: bool = False) -> None:
        verify_type("value", value, str)
        self.assign(File(file_path=value), wait)

    def save_files(self, value: Union[str, Iterable[str]], wait: bool = False) -> None:
        if is_collection(value):
            verify_collection_type("value", value, str)
        else:
            verify_type("value", value, str)

        with self._experiment.lock():
            attr = self._experiment.get_attribute(self._path)
            if not attr:
                attr = FileSet(self._experiment, parse_path(self._path))
                attr.save_files(value, wait)
                self._experiment.set_attribute(self._path, attr)
            else:
                attr.save_files(value, wait)

    def log(self,
            value: Union[int, float, str, Image, Iterable[int], Iterable[float], Iterable[str], Iterable[Image]],
            step=None,
            timestamp=None,
            wait: bool = False) -> None:
        verify_type("step", step, (int, float, type(None)))
        verify_type("timestamp", timestamp, (int, float, type(None)))

        with self._experiment.lock():
            attr = self._experiment.get_attribute(self._path)
            if not attr:
                if is_collection(value):
                    if value:
                        first_value = next(iter(value))
                    else:
                        raise ValueError("Cannot deduce value type: `value` cannot be empty")
                else:
                    first_value = value

                if is_float(first_value):
                    attr = FloatSeries(self._experiment, parse_path(self._path))
                elif is_string(first_value):
                    attr = StringSeries(self._experiment, parse_path(self._path))
                elif isinstance(first_value, Image):
                    attr = ImageSeries(self._experiment, parse_path(self._path))
                elif is_float_like(first_value):
                    attr = FloatSeries(self._experiment, parse_path(self._path))
                elif is_string_like(first_value):
                    attr = StringSeries(self._experiment, parse_path(self._path))
                else:
                    raise TypeError("Value of unsupported type {}".format(type(first_value)))

                attr.log(value, step=step, timestamp=timestamp, wait=wait)
                self._experiment.set_attribute(self._path, attr)
            else:
                attr.log(value, step=step, timestamp=timestamp, wait=wait)

    def add(self, values: Union[str, Iterable[str]], wait: bool = False) -> None:
        verify_type("values", values, (str, Iterable))
        with self._experiment.lock():
            attr = self._experiment.get_attribute(self._path)
            if not attr:
                attr = StringSet(self._experiment, parse_path(self._path))
                attr.add(values, wait)
                self._experiment.set_attribute(self._path, attr)
            else:
                attr.add(values, wait)

    def pop(self, path: str, wait: bool = False) -> None:
        verify_type("path", path, str)
        self._experiment.pop(join_paths(self._path, path), wait)

    def __delitem__(self, path) -> None:
        self.pop(path)
