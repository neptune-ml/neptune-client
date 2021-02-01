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

# pylint: disable=protected-access
from typing import Optional, Iterable

from mock import MagicMock, call, patch

from neptune.alpha.internal.operation import ClearFloatLog, LogFloats, Operation, ConfigFloatSeries, ClearStringLog
from neptune.alpha.attributes.series.float_series import FloatSeries, FloatSeriesVal
from neptune.alpha.attributes.series.string_series import StringSeries, StringSeriesVal
from neptune.alpha.attributes.series.series import Series

from tests.neptune.alpha.attributes.test_attribute_base import TestAttributeBase


@patch("time.time", new=TestAttributeBase._now)
class TestSeries(TestAttributeBase):

    def test_assign(self):
        value = FloatSeriesVal([17, 3.6], min=0, max=100, unit="%")
        expected = [LogFloats.ValueType(17, None, self._now()), LogFloats.ValueType(3.6, None, self._now())]

        processor = MagicMock()
        exp, path, wait = self._create_experiment(processor), self._random_path(), self._random_wait()
        var = FloatSeries(exp, path)
        var.assign(value, wait=wait)
        self.assertEqual(3, processor.enqueue_operation.call_count)
        processor.enqueue_operation.assert_has_calls([
            call(ConfigFloatSeries(path, min=0, max=100, unit="%"), False),
            call(ClearFloatLog(path), False),
            call(LogFloats(path, expected), wait)
        ])

    def test_assign_empty(self):
        processor = MagicMock()
        exp, path, wait = self._create_experiment(processor), self._random_path(), self._random_wait()
        var = StringSeries(exp, path)
        var.assign(StringSeriesVal([]), wait=wait)
        processor.enqueue_operation.assert_called_once_with(ClearStringLog(path), wait)

    def test_log(self):
        value_and_expected = [
            (13, [LogFloats.ValueType(13, None, self._now())]),
            (15.3, [LogFloats.ValueType(15.3, None, self._now())]),
            ([], []),
            ([1, 9, 7], [LogFloats.ValueType(1, None, self._now()),
                         LogFloats.ValueType(9, None, self._now()),
                         LogFloats.ValueType(7, None, self._now())]),
            ((1, 9, 7), [LogFloats.ValueType(1, None, self._now()),
                         LogFloats.ValueType(9, None, self._now()),
                         LogFloats.ValueType(7, None, self._now())]),
            ({1, 9, 7}, [LogFloats.ValueType(1, None, self._now()),
                         LogFloats.ValueType(9, None, self._now()),
                         LogFloats.ValueType(7, None, self._now())])
        ]

        for value, expected in value_and_expected:
            processor = MagicMock()
            exp, path, wait = self._create_experiment(processor), self._random_path(), self._random_wait()
            var = FloatSeries(exp, path)
            var.log(value, wait=wait)
            processor.enqueue_operation.assert_called_once_with(LogFloats(path, expected), wait)

    def test_log_with_step(self):
        value_step_and_expected = [
            (13, 5.3, LogFloats.ValueType(13, 5.3, self._now())),
            (15.3, 10, LogFloats.ValueType(15.3, 10, self._now())),
            ([13], 5.3, LogFloats.ValueType(13, 5.3, self._now())),
            ((13,), 5.3, LogFloats.ValueType(13, 5.3, self._now())),
            ({13}, 5.3, LogFloats.ValueType(13, 5.3, self._now()))
        ]

        for value, step, expected in value_step_and_expected:
            processor = MagicMock()
            exp, path, wait = self._create_experiment(processor), self._random_path(), self._random_wait()
            var = FloatSeries(exp, path)
            var.log(value, step=step, wait=wait)
            processor.enqueue_operation.assert_called_once_with(LogFloats(path, [expected]), wait)

    def test_log_with_timestamp(self):
        value_step_and_expected = [
            (13, 5.3, LogFloats.ValueType(13, None, 5.3)),
            (15.3, 10, LogFloats.ValueType(15.3, None, 10))
        ]

        for value, ts, expected in value_step_and_expected:
            processor = MagicMock()
            exp, path, wait = self._create_experiment(processor), self._random_path(), self._random_wait()
            var = FloatSeries(exp, path)
            var.log(value, timestamp=ts, wait=wait)
            processor.enqueue_operation.assert_called_once_with(LogFloats(path, [expected]), wait)

    def test_log_value_errors(self):
        processor = MagicMock()
        exp, path = self._create_experiment(processor), self._random_path()
        attr = FloatSeries(exp, path)

        with self.assertRaises(ValueError):
            attr.log(["str", 5])
        with self.assertRaises(ValueError):
            attr.log([5, 10], step=10)
        with self.assertRaises(TypeError):
            attr.log(5, step="str")
        with self.assertRaises(TypeError):
            attr.log(5, timestamp="str")

    def test_clear(self):
        processor = MagicMock()
        exp, path, wait = self._create_experiment(processor), self._random_path(), self._random_wait()
        var = FloatSeries(exp, path)
        var.clear(wait=wait)
        processor.enqueue_operation.assert_called_once_with(ClearFloatLog(path), wait)

    class SeriesTestClass(Series[FloatSeriesVal, int]):

        def _get_log_operation_from_value(self,
                                          value: FloatSeriesVal,
                                          step: Optional[float],
                                          timestamp: float) -> Operation:
            values = [LogFloats.ValueType(val, step=step, ts=timestamp) for val in value.values]
            return LogFloats(self._path, values)

        def _get_log_operation_from_data(self,
                                         data_list: Iterable[int],
                                         step: Optional[float],
                                         timestamp: float) -> Operation:
            return LogFloats(self._path, [LogFloats.ValueType(data_list, step, timestamp)])

        def _get_clear_operation(self) -> Operation:
            return ClearFloatLog(self._path)

        def _data_to_value(self, value: Iterable) -> FloatSeriesVal:
            return FloatSeriesVal(value)

        def _is_value_type(self, value) -> bool:
            return isinstance(value, FloatSeriesVal)
