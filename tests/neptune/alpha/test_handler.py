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
import os
import unittest
# pylint: disable=protected-access
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from tempfile import TemporaryDirectory, NamedTemporaryFile
from unittest.mock import patch

import PIL
from neptune.alpha.attributes.series import ImageSeries

from neptune.alpha import init, ANONYMOUS
from neptune.alpha.attributes.atoms.datetime import Datetime
from neptune.alpha.attributes.atoms.file import File
from neptune.alpha.attributes.atoms.float import Float
from neptune.alpha.attributes.atoms.string import String
from neptune.alpha.attributes.sets.string_set import StringSet
from neptune.alpha.envs import PROJECT_ENV_NAME, API_TOKEN_ENV_NAME
from neptune.alpha.types import Image
from neptune.alpha.types.atoms.datetime import Datetime as DatetimeVal
from neptune.alpha.types.atoms.float import Float as FloatVal
from neptune.alpha.types.atoms.string import String as StringVal
from neptune.alpha.types.series.image_series import ImageSeries as ImageSeriesVal
from neptune.alpha.types.series.float_series import FloatSeries as FloatSeriesVal
from neptune.alpha.types.series.string_series import StringSeries as StringSeriesVal
from neptune.alpha.types.sets.string_set import StringSet as StringSetVal


class TestHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ[PROJECT_ENV_NAME] = "organization/project"
        os.environ[API_TOKEN_ENV_NAME] = ANONYMOUS

    def test_set(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        now = datetime.now()
        exp['some/num/val'] = 5
        exp['some/str/val'] = "some text"
        exp['some/datetime/val'] = now
        exp.wait()
        self.assertEqual(exp['some/num/val'].get(), 5)
        self.assertEqual(exp['some/str/val'].get(), "some text")
        self.assertEqual(exp['some/datetime/val'].get(), now.replace(microsecond=1000*int(now.microsecond/1000)))
        self.assertIsInstance(exp.get_structure()['some']['num']['val'], Float)
        self.assertIsInstance(exp.get_structure()['some']['str']['val'], String)
        self.assertIsInstance(exp.get_structure()['some']['datetime']['val'], Datetime)

    def test_assign_atom(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        now = datetime.now()
        exp['some/num/val'].assign(5)
        exp['some/str/val'].assign("some text", wait=True)
        exp['some/datetime/val'].assign(now)
        self.assertEqual(exp['some/num/val'].get(), 5)
        self.assertEqual(exp['some/str/val'].get(), "some text")
        self.assertEqual(exp['some/datetime/val'].get(), now.replace(microsecond=1000*int(now.microsecond/1000)))
        self.assertIsInstance(exp.get_structure()['some']['num']['val'], Float)
        self.assertIsInstance(exp.get_structure()['some']['str']['val'], String)
        self.assertIsInstance(exp.get_structure()['some']['datetime']['val'], Datetime)

        now = now + timedelta(seconds=3, microseconds=500000)
        exp['some/num/val'].assign(FloatVal(15))
        exp['some/str/val'].assign(StringVal("other text"), wait=False)
        exp['some/datetime/val'].assign(DatetimeVal(now), wait=True)
        self.assertEqual(exp['some/num/val'].get(), 15)
        self.assertEqual(exp['some/str/val'].get(), "other text")
        self.assertEqual(exp['some/datetime/val'].get(), now.replace(microsecond=1000*int(now.microsecond/1000)))
        self.assertIsInstance(exp.get_structure()['some']['num']['val'], Float)
        self.assertIsInstance(exp.get_structure()['some']['str']['val'], String)
        self.assertIsInstance(exp.get_structure()['some']['datetime']['val'], Datetime)

    @patch('neptune.alpha.internal.backends.neptune_backend_mock.copyfile')
    def test_save_download_file(self, copy_mock):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/num/file'].save("path/to/other/file.txt")
        self.assertIsInstance(exp.get_structure()['some']['num']['file'], File)

        exp['some/num/file'].download("path/to/download/alternative_file.txt")
        copy_mock.assert_called_with(
            os.path.abspath("path/to/other/file.txt"), os.path.abspath("path/to/download/alternative_file.txt"))
        copy_mock.reset_mock()

        exp['some/num/file'].download()
        copy_mock.assert_called_with(
            os.path.abspath("path/to/other/file.txt"), os.path.abspath("file.txt"))
        copy_mock.reset_mock()

        exp['some/num/file'].download("path/to/other/file.txt")
        copy_mock.assert_not_called()

        with self.assertRaises(TypeError):
            exp['some/num/file'].download(123)

    def test_save_download_text_stream_to_given_destination(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        data = "Some test content of the stream"

        exp['some/num/file'] = StringIO(data)
        self.assertIsInstance(exp.get_structure()['some']['num']['file'], File)

        with NamedTemporaryFile("w") as temp_file:
            exp['some/num/file'].download(temp_file.name)
            with open(temp_file.name, "rt") as file:
                self.assertEqual(file.read(), data)

    def test_save_download_binary_stream_to_default_destination(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        data = b"Some test content of the stream"

        exp['some/num/file'] = BytesIO(data)
        self.assertIsInstance(exp.get_structure()['some']['num']['file'], File)

        exp['some/num/file'].download()
        with TemporaryDirectory() as temp_dir:
            with patch('neptune.alpha.internal.backends.neptune_backend_mock.os.path.abspath') as abspath_mock:
                abspath_mock.side_effect = lambda path: os.path.normpath(temp_dir + path)
                exp['some/num/file'].download()
            with open(temp_dir + 'stream.bin', "rb") as file:
                self.assertEqual(file.read(), data)

    @patch('neptune.alpha.internal.utils.glob',
           new=lambda path: [path.replace('*', 'file.txt')])
    @patch('neptune.alpha.internal.backends.neptune_backend_mock.ZipFile.write')
    def test_save_files_download_zip(self, zip_write_mock):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/artifacts'].save_files("path/to/file.txt")
        exp['some/artifacts'].save_files("path/to/other/*")

        with TemporaryDirectory() as temp_dir:
            exp['some/artifacts'].download_zip(temp_dir)
            exp['some/artifacts'].download_zip(temp_dir)

        zip_write_mock.assert_any_call(os.path.abspath("path/to/file.txt"), "path/to/file.txt")
        zip_write_mock.assert_any_call(os.path.abspath("path/to/other/file.txt"), "path/to/other/file.txt")

    def test_assign_series(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/num/val'].assign(FloatSeriesVal([1, 2, 0, 10]))
        exp['some/str/val'].assign(StringSeriesVal(["text1", "text2"]), wait=True)
        exp['some/img/val'].assign(ImageSeriesVal([Image(PIL.Image.new('RGB', (10, 15), color='red'))]))
        self.assertEqual(exp['some']['num']['val'].get_last(), 10)
        self.assertEqual(exp['some']['str']['val'].get_last(), "text2")
        self.assertIsInstance(exp.get_structure()['some']['img']['val'], ImageSeries)

        exp['some/num/val'].assign(FloatSeriesVal([122, 543, 2, 5]))
        exp['some/str/val'].assign(StringSeriesVal(["other 1", "other 2", "other 3"]), wait=True)
        self.assertEqual(exp['some']['num']['val'].get_last(), 5)
        self.assertEqual(exp['some']['str']['val'].get_last(), "other 3")

    def test_log(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/num/val'].log(5)
        exp['some/str/val'].log("some text")
        exp['some/img/val'].log(Image(PIL.Image.new('RGB', (60, 30), color='red')))
        self.assertEqual(exp['some']['num']['val'].get_last(), 5)
        self.assertEqual(exp['some']['str']['val'].get_last(), "some text")
        self.assertIsInstance(exp.get_structure()['some']['img']['val'], ImageSeries)

    def test_log_many_values(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/num/val'].log([5, 10, 15])
        exp['some/str/val'].log(["some text", "other"])
        exp['some/img/val'].log([Image(PIL.Image.new('RGB', (60, 30), color='red')),
                                 Image(PIL.Image.new('RGB', (20, 90), color='red'))])
        self.assertEqual(exp['some']['num']['val'].get_last(), 15)
        self.assertEqual(exp['some']['str']['val'].get_last(), "other")
        self.assertIsInstance(exp.get_structure()['some']['img']['val'], ImageSeries)

    def test_log_value_errors(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        img = Image(PIL.Image.new('RGB', (60, 30), color='red'))

        with self.assertRaises(ValueError):
            exp['x'].log([])
        with self.assertRaises(TypeError):
            exp['x'].log(["str", 5])
        with self.assertRaises(ValueError):
            exp['x'].log([5, 10], step=10)

        exp['some/num/val'].log([5], step=1)
        exp['some/num/val'].log([])
        with self.assertRaises(TypeError):
            exp['some/num/val'].log("str")
        with self.assertRaises(TypeError):
            exp['some/num/val'].log(img)

        exp['some/str/val'].log(["str"], step=1)
        exp['some/str/val'].log([])
        with self.assertRaises(TypeError):
            exp['some/str/val'].log(5)
        with self.assertRaises(TypeError):
            exp['some/str/val'].log(img)

        exp['some/img/val'].log([img], step=1)
        exp['some/img/val'].log([])
        with self.assertRaises(TypeError):
            exp['some/img/val'].log(5)
        with self.assertRaises(TypeError):
            exp['some/img/val'].log("str")

        self.assertEqual(exp['some']['num']['val'].get_last(), 5)
        self.assertEqual(exp['some']['str']['val'].get_last(), "str")
        self.assertIsInstance(exp.get_structure()['some']['img']['val'], ImageSeries)

    def test_assign_set(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/str/val'].assign(StringSetVal(["tag1", "tag2"]), wait=True)
        self.assertEqual(exp['some/str/val'].get(), {"tag1", "tag2"})
        self.assertIsInstance(exp.get_structure()['some']['str']['val'], StringSet)

        exp['some/str/val'].assign(StringSetVal(["other_1", "other_2", "other_3"]), wait=True)
        self.assertEqual(exp['some/str/val'].get(), {"other_1", "other_2", "other_3"})
        self.assertIsInstance(exp.get_structure()['some']['str']['val'], StringSet)

    def test_add(self):
        exp = init(connection_mode="debug", flush_period=0.5)

        exp['some/str/val'].add(["some text", "something else"], wait=True)
        self.assertEqual(exp['some/str/val'].get(), {"some text", "something else"})

        exp['some/str/val'].add("one more", wait=True)
        self.assertEqual(exp['some/str/val'].get(), {"some text", "something else", "one more"})

        self.assertIsInstance(exp.get_structure()['some']['str']['val'], StringSet)

    def test_pop(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/num/val'].assign(3, wait=True)
        self.assertIn('some', exp.get_structure())
        ns = exp['some']
        ns.pop('num/val', wait=True)
        self.assertNotIn('some', exp.get_structure())

    def test_del(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        exp['some/num/val'].assign(3)
        self.assertIn('some', exp.get_structure())
        ns = exp['some']
        del ns['num/val']
        self.assertNotIn('some', exp.get_structure())

    def test_lookup(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        ns = exp['some/ns']
        ns['val'] = 5
        exp.wait()
        self.assertEqual(exp['some/ns/val'].get(), 5)

        ns = exp['other/ns']
        exp['other/ns/some/value'] = 3
        exp.wait()
        self.assertEqual(ns['some/value'].get(), 3)

    def test_attribute_error(self):
        exp = init(connection_mode="debug", flush_period=0.5)
        with self.assertRaises(AttributeError):
            exp['var'].something()

    def test_float_like_types(self):
        exp = init(connection_mode="debug", flush_period=0.5)

        exp.define("attr1", self.FloatLike(5))
        self.assertEqual(exp['attr1'].get(), 5)
        exp["attr1"] = "234"
        self.assertEqual(exp['attr1'].get(), 234)
        with self.assertRaises(ValueError):
            exp["attr1"] = "234a"

        exp["attr2"].assign(self.FloatLike(34))
        self.assertEqual(exp['attr2'].get(), 34)
        exp["attr2"].assign("555")
        self.assertEqual(exp['attr2'].get(), 555)
        with self.assertRaises(ValueError):
            exp["attr2"].assign("string")

        exp["attr3"].log(self.FloatLike(34))
        self.assertEqual(exp['attr3'].get_last(), 34)
        exp["attr3"].log(["345", self.FloatLike(34), 4, 13.])
        self.assertEqual(exp['attr3'].get_last(), 13)
        with self.assertRaises(ValueError):
            exp["attr3"].log([4, "234a"])

    def test_string_like_types(self):
        exp = init(connection_mode="debug", flush_period=0.5)

        exp["attr1"] = "234"
        exp["attr1"] = self.FloatLike(12356)
        self.assertEqual(exp['attr1'].get(), "TestHandler.FloatLike(value=12356)")

        exp["attr2"].log("xxx")
        exp["attr2"].log(["345", self.FloatLike(34), 4, 13.])
        self.assertEqual(exp['attr2'].get_last(), "13.0")

    @dataclass
    class FloatLike:

        value: float

        def __float__(self):
            return float(self.value)
