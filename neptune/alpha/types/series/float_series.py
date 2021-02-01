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

from typing import TypeVar, TYPE_CHECKING, Optional, Union

from neptune.alpha.internal.utils import is_collection
from neptune.alpha.types.series.series import Series

if TYPE_CHECKING:
    from neptune.alpha.types.value_visitor import ValueVisitor

Ret = TypeVar('Ret')


class FloatSeries(Series):

    # pylint: disable=redefined-builtin
    def __init__(self,
                 values,
                 min: Optional[Union[float, int]] = None,
                 max: Optional[Union[float, int]] = None,
                 unit: Optional[str] = None):
        if not is_collection(values):
            raise TypeError("`values` is not a collection")
        self._values = [float(value) for value in values]
        self._min = min
        self._max = max
        self._unit = unit

    def accept(self, visitor: 'ValueVisitor[Ret]') -> Ret:
        return visitor.visit_float_series(self)

    @property
    def values(self):
        return self._values

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def unit(self):
        return self._unit

    def __str__(self):
        return "FloatSeries({})".format(str(self.values))
