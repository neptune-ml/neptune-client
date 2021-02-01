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
from typing import Union


from neptune.alpha.internal.operation import AssignString
from neptune.alpha.types.atoms.string import String as StringVal
from neptune.alpha.attributes.atoms.atom import Atom


class String(Atom):

    def assign(self, value: Union[StringVal, str], wait: bool = False):
        if not isinstance(value, StringVal):
            value = StringVal(value)

        with self._experiment.lock():
            self._enqueue_operation(AssignString(self._path, value.value), wait)

    def get(self) -> str:
        # pylint: disable=protected-access
        val = self._backend.get_string_attribute(self._experiment_uuid, self._path)
        return val.value
