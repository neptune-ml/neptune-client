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

from typing import TypeVar, TYPE_CHECKING

from neptune.alpha.internal.utils import is_collection
from neptune.alpha.types.series.series import Series

if TYPE_CHECKING:
    from neptune.alpha.types.value_visitor import ValueVisitor

Ret = TypeVar('Ret')


class StringSeries(Series):

    def __init__(self, values):
        if not is_collection(values):
            raise TypeError("`values` is not a collection")
        self._values = [str(value) for value in values]

    def accept(self, visitor: 'ValueVisitor[Ret]') -> Ret:
        return visitor.visit_string_series(self)

    @property
    def values(self):
        return self._values

    def __str__(self):
        return "StringSeries({})".format(str(self.values))
