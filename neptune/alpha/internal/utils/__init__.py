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
import base64
from io import IOBase, StringIO, BytesIO
import logging
import os
from glob import glob
from typing import Union, TypeVar, Iterable, List, Set, Optional, Any
from neptune.internal.hardware.constants import BYTES_IN_ONE_MB

T = TypeVar('T')

_logger = logging.getLogger(__name__)


def replace_patch_version(version: str):
    return version[:version.index(".", version.index(".") + 1)] + ".0"


def verify_type(var_name: str, var, expected_type: Union[type, tuple]):
    try:
        if isinstance(expected_type, tuple):
            type_name = " or ".join(get_type_name(t) for t in expected_type)
        else:
            type_name = get_type_name(expected_type)
    except Exception as e:
        # Just to be sure that nothing weird will be raised here
        raise TypeError("Incorrect type of {}".format(var_name)) from e

    if not isinstance(var, expected_type):
        raise TypeError("{} must be a {} (was {})".format(var_name, type_name, type(var)))

    if isinstance(var, IOBase) and not hasattr(var, 'read'):
        raise TypeError("{} is a stream, which does not implement read method".format(var_name))


def is_stream(var):
    return isinstance(var, IOBase) and hasattr(var, 'read')


def is_float(var):
    return isinstance(var, (float, int))


def is_string(var):
    return isinstance(var, str)


def is_float_like(var):
    try:
        _ = float(var)
        return True
    except (ValueError, TypeError):
        return False


def is_string_like(var):
    try:
        _ = str(var)
        return True
    except ValueError:
        return False


def get_type_name(_type: Union[type, tuple]):
    return _type.__name__ if hasattr(_type, '__name__') else str(_type)


def verify_collection_type(var_name: str, var, expected_type: Union[type, tuple]):
    verify_type(var_name, var, (list, set, tuple))
    for value in var:
        verify_type("elements of collection '{}'".format(var_name), value, expected_type)


def is_collection(var) -> bool:
    return isinstance(var, (list, set, tuple))


def base64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode('utf-8')


def base64_decode(data: str) -> bytes:
    return base64.b64decode(data.encode('utf-8'))


def get_absolute_paths(file_globs: Iterable[str]) -> List[str]:
    expanded_paths: Set[str] = set()
    for file_glob in file_globs:
        expanded_paths |= set(glob(file_glob))
    return list(os.path.abspath(expanded_file) for expanded_file in expanded_paths)


def get_common_root(absolute_paths: List[str]) -> Optional[str]:
    try:
        common_root = os.path.commonpath(absolute_paths)
        if os.path.isfile(common_root):
            common_root = os.path.dirname(common_root)
        if common_root.startswith(os.getcwd() + os.sep):
            common_root = os.getcwd()
        return common_root
    except ValueError:
        return None


STREAM_SIZE_LIMIT_MB = 15


def get_stream_content(stream: IOBase) -> (Optional[str], str):
    if stream.seekable():
        stream.seek(0)

    content = stream.read(STREAM_SIZE_LIMIT_MB * 1024 * 1024 + 1)
    default_name = "stream.txt" if isinstance(content, str) else "stream.bin"
    if isinstance(content, str):
        content = content.encode('utf-8')

    if len(content) > STREAM_SIZE_LIMIT_MB * 1024 * 1024:
        _logger.warning('Your stream is larger than %dMB. Neptune supports saving files smaller than %dMB.',
                        STREAM_SIZE_LIMIT_MB, STREAM_SIZE_LIMIT_MB)
        return None, default_name

    while True:
        chunk = stream.read(1024 * 1024)
        if chunk is None:
            continue
        elif not chunk:
            break
        else:
            if not isinstance(content, BytesIO):
                content = BytesIO(content)
                content.seek(0, 2)
            if isinstance(chunk, str):
                chunk = chunk.encode('utf-8')
            if content.tell() + len(chunk) > STREAM_SIZE_LIMIT_MB * 1024 * 1024:
                _logger.warning('Your stream is larger than %dMB. Neptune supports saving files smaller than %dMB.',
                                STREAM_SIZE_LIMIT_MB, STREAM_SIZE_LIMIT_MB)
                return None, default_name
            content.write(chunk)
    if isinstance(content, BytesIO):
        content = content.getvalue()

    return base64_encode(content), default_name
