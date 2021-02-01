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

from typing import Optional, Iterable


from neptune.alpha.types.series.string_series import StringSeries as StringSeriesVal

from neptune.alpha.internal.operation import LogStrings, ClearStringLog, Operation
from neptune.alpha.attributes.series.series import Series

Val = StringSeriesVal
Data = str


class StringSeries(Series[Val, Data]):

    def _get_log_operation_from_value(self, value: Val, step: Optional[float], timestamp: float) -> Operation:
        values = [LogStrings.ValueType(val, step=step, ts=timestamp) for val in value.values]
        return LogStrings(self._path, values)

    def _get_log_operation_from_data(self,
                                     data_list: Iterable[Data],
                                     step: Optional[float],
                                     timestamp: float) -> Operation:
        return LogStrings(self._path, [LogStrings.ValueType(data, step, timestamp) for data in data_list])

    def _get_clear_operation(self) -> Operation:
        return ClearStringLog(self._path)

    def _data_to_value(self, value: Iterable) -> Val:
        return StringSeriesVal(value)

    def _is_value_type(self, value) -> bool:
        return isinstance(value, StringSeriesVal)

    def get_last(self) -> str:
        # pylint: disable=protected-access
        val = self._backend.get_string_series_attribute(self._experiment_uuid, self._path)
        return val.last
