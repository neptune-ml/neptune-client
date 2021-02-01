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

from mock import patch, MagicMock

from neptune.alpha.attributes.series.image_series import ImageSeries

from tests.neptune.alpha.attributes.test_attribute_base import TestAttributeBase


@patch("time.time", new=TestAttributeBase._now)
class TestStringSeries(TestAttributeBase):

    def test_assign_type_error(self):
        values = [[5.], ["text"], 55, "string", None]
        for value in values:
            with self.assertRaises(Exception):
                ImageSeries(MagicMock(), MagicMock()).assign(value)

    def test_log_type_error(self):
        values = [[5.], ["text"], [[]], 55, None]
        for value in values:
            with self.assertRaises(TypeError):
                ImageSeries(MagicMock(), MagicMock()).log(value)
