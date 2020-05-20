#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
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
import logging
import os
import re
import time
import traceback

import pandas as pd
import requests
import six
from pandas.errors import EmptyDataError

from neptune.api_exceptions import ExperimentAlreadyFinished, ChannelDoesNotExist, PathInProjectNotFound
from neptune.exceptions import FileNotFound, InvalidChannelValue, NoChannelValue, NotADirectory
from neptune.internal.channels.channels import ChannelValue, ChannelType, ChannelNamespace
from neptune.internal.channels.channels_values_sender import ChannelsValuesSender
from neptune.internal.execution.execution_context import ExecutionContext
from neptune.internal.storage.storage_utils import upload_to_storage, UploadEntry, normalize_file_name
from neptune.internal.utils.image import get_image_content
from neptune.utils import align_channels_on_x, is_float, is_nan_or_inf

_logger = logging.getLogger(__name__)


# pylint: disable=too-many-lines
class Experiment(object):
    """A class for managing Neptune experiment.

    Each time User creates new experiment instance of this class is created.
    It lets you manage experiment, :meth:`~neptune.experiments.Experiment.log_metric`,
    :meth:`~neptune.experiments.Experiment.log_text`,
    :meth:`~neptune.experiments.Experiment.log_image`,
    :meth:`~neptune.experiments.Experiment.set_property`,
    and much more.


    Args:
        backend (:obj:`neptune.Backend`): A Backend object
        project (:obj:`neptune.Project`): The project this experiment belongs to
        _id (:obj:`str`): Experiment id
        internal_id (:obj:`str`): internal UUID

    Example:
        Assuming that `project` is an instance of :class:`~neptune.projects.Project`.

        .. code:: python3

            experiment = project.create_experiment()

    Warning:
        User should never create instances of this class manually.
        Always use: :meth:`~neptune.projects.Project.create_experiment`.

    """

    IMAGE_SIZE_LIMIT_MB = 15

    def __init__(self, backend, project, _id, internal_id):
        self._backend = backend
        self._project = project
        self._id = _id
        self._internal_id = internal_id
        self._channels_values_sender = ChannelsValuesSender(self)
        self._execution_context = ExecutionContext(backend, self)

    @property
    def id(self):
        """Experiment short id

        | Combination of project key and unique experiment number.
        | Format is ``<project_key>-<experiment_number>``, for example: ``MPI-142``.

        Returns:
            :obj:`str` - experiment short id

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                exp_id = experiment.id

        """
        return self._id

    @property
    def name(self):
        """Experiment name

        Returns:
            :obj:`str` experiment name

        Examples:
            Assuming that `project` is an instance of :class:`~neptune.projects.Project`.

            .. code:: python3

                experiment = project.create_experiment('exp_name')
                exp_name = experiment.name
        """
        return self._backend.get_experiment(self._internal_id).name

    @property
    def state(self):
        """Current experiment state

        Possible values: `'running'`, `'succeeded'`, `'failed'`, `'aborted'`.

        Returns:
            :obj:`str` - current experiment state

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                state_str = experiment.state
        """
        return self._backend.get_experiment(self._internal_id).state

    @property
    def internal_id(self):
        return self._internal_id

    @property
    def limits(self):
        return {
            'channels': {
                'numeric': 1000,
                'text': 100,
                'image': 100
            }
        }

    def get_system_properties(self):
        """Retrieve experiment properties.

        | Experiment properties are for example: `owner`, `created`, `name`, `hostname`.
        | List of experiment properties may change over time.

        Returns:
            :obj:`dict` - dictionary mapping a property name to value.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                sys_properties = experiment.get_system_properties
        """
        experiment = self._backend.get_experiment(self._internal_id)
        return {
            'id': experiment.shortId,
            'name': experiment.name,
            'created': experiment.timeOfCreation,
            'finished': experiment.timeOfCompletion,
            'running_time': experiment.runningTime,
            'owner': experiment.owner,
            'storage_size': experiment.storageSize,
            'channels_size': experiment.channelsSize,
            'size': experiment.storageSize + experiment.channelsSize,
            'tags': experiment.tags,
            'notes': experiment.description,
            'description': experiment.description,
            'hostname': experiment.hostname
        }

    def get_tags(self):
        """Get tags associated with experiment.

        Returns:
            :obj:`list` of :obj:`str` with all tags for this experiment.

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                experiment.get_tags()
        """
        return self._backend.get_experiment(self._internal_id).tags

    def append_tag(self, tag, *tags):
        """Append tag(s) to the current experiment.

        Alias: :meth:`~neptune.experiments.Experiment.append_tags`.
        Only ``[a-zA-Z0-9]`` and ``-`` (dash) characters are allowed in tags.

        Args:
            tag (single :obj:`str` or multiple :obj:`str` or :obj:`list` of :obj:`str`):
                Tag(s) to add to the current experiment.

                    * If :obj:`str` is passed, singe tag is added.
                    * If multiple - comma separated - :obj:`str` are passed, all of them are added as tags.
                    * If :obj:`list` of :obj:`str` is passed, all elements of the :obj:`list` are added as tags.

        Examples:

            .. code:: python3

                neptune.append_tag('new-tag')  # single tag
                neptune.append_tag('first-tag', 'second-tag', 'third-tag')  # few str
                neptune.append_tag(['first-tag', 'second-tag', 'third-tag'])  # list of str
        """
        if isinstance(tag, list):
            tags_list = tag
        else:
            tags_list = [tag] + list(tags)
        self._backend.update_tags(experiment=self,
                                  tags_to_add=tags_list,
                                  tags_to_delete=[])

    def append_tags(self, tag, *tags):
        """Append tag(s) to the current experiment.

        Alias for: :meth:`~neptune.experiments.Experiment.append_tag`
        """
        self.append_tag(tag, *tags)

    def remove_tag(self, tag):
        """Removes single tag from the experiment.

        Args:
            tag (:obj:`str`): Tag to be removed

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                # assuming experiment has tags: `['tag-1', 'tag-2']`.
                experiment.remove_tag('tag-1')

        Note:
            Removing a tag that is not assigned to this experiment is silently ignored.
        """
        self._backend.update_tags(experiment=self,
                                  tags_to_add=[],
                                  tags_to_delete=[tag])

    def get_channels(self):
        """Alias for :meth:`~neptune.experiments.Experiment.get_logs`
        """
        return self.get_logs()

    def get_logs(self):
        """Retrieve all log names along with their last values for this experiment.

        Returns:
            :obj:`dict` - A dictionary mapping a log names to the log's last value.

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                exp_logs = experiment.get_logs()
        """
        experiment = self._backend.get_experiment(self.internal_id)
        channels_last_values_by_name = dict((ch.channelName, ch) for ch in experiment.channelsLastValues)
        channels = dict()
        for ch in experiment.channels:
            last_value = channels_last_values_by_name.get(ch.name, None)
            if last_value is not None:
                ch.x = last_value.x
                ch.y = last_value.y
            elif ch.lastX is not None:
                ch.x = ch.lastX
                ch.y = None
            else:
                ch.x = None
                ch.y = None
            channels[ch.name] = ch
        return channels

    def _get_system_channels(self):
        channels = self._backend.get_system_channels(self)
        return dict((ch.name, ch) for ch in channels)

    def send_metric(self, channel_name, x, y=None, timestamp=None):
        """Log metrics (numeric values) in Neptune.

        Alias for :meth:`~neptune.experiments.Experiment.log_metric`
        """
        return self.log_metric(channel_name, x, y, timestamp)

    def log_metric(self, log_name, x, y=None, timestamp=None):
        """Log metrics (numeric values) in Neptune

        | If a log with provided ``log_name`` does not exist, it is created automatically.
        | If log exists (determined by ``log_name``), then new value is appended to it.

        Args:
            log_name (:obj:`str`): The name of log, i.e. `mse`, `loss`, `accuracy`.
            x (:obj:`double`): Depending, whether ``y`` parameter is passed:

                * ``y`` not passed: The value of the log (data-point).
                * ``y`` passed: Index of log entry being appended. Must be strictly increasing.

            y (:obj:`double`, optional, default is ``None``): The value of the log (data-point).
            timestamp (:obj:`time`, optional, default is ``None``):
                Timestamp to be associated with log entry. Must be Unix time.
                If ``None`` is passed, `time.time() <https://docs.python.org/3.6/library/time.html#time.time>`_
                (Python 3.6 example) is invoked to obtain timestamp.

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment` and
            'accuracy' log does not exists:

            .. code:: python3

                # Both calls below have the same effect

                # Common invocation, providing log name and value
                experiment.log_metric('accuracy', 0.5)
                experiment.log_metric('accuracy', 0.65)
                experiment.log_metric('accuracy', 0.8)

                # Providing both x and y params
                experiment.log_metric('accuracy', 0, 0.5)
                experiment.log_metric('accuracy', 1, 0.65)
                experiment.log_metric('accuracy', 2, 0.8)

        Note:
            For efficiency, logs are uploaded in batches via a queue.
            Hence, if you log a lot of data, you may experience slight delays in Neptune web application.
        Note:
            Passing either ``x`` or ``y`` coordinate as NaN or +/-inf causes this log entry to be ignored.
            Warning is printed to ``stdout``.
        """
        x, y = self._get_valid_x_y(x, y)

        if not is_float(y):
            raise InvalidChannelValue(expected_type='float', actual_type=type(y).__name__)

        value = ChannelValue(x, dict(numeric_value=y), timestamp)
        self._channels_values_sender.send(log_name, ChannelType.NUMERIC.value, value)

    def send_text(self, channel_name, x, y=None, timestamp=None):
        """Log text data in Neptune.

        Alias for :meth:`~neptune.experiments.Experiment.log_text`
        """
        return self.log_text(channel_name, x, y, timestamp)

    def log_text(self, log_name, x, y=None, timestamp=None):
        """Log text data in Neptune

        | If a log with provided ``log_name`` does not exist, it is created automatically.
        | If log exists (determined by ``log_name``), then new value is appended to it.

        Args:
            log_name (:obj:`str`): The name of log, i.e. `mse`, `my_text_data`, `timing_info`.
            x (:obj:`double` or :obj:`str`): Depending, whether ``y`` parameter is passed:

                * ``y`` not passed: The value of the log (data-point). Must be ``str``.
                * ``y`` passed: Index of log entry being appended. Must be strictly increasing.

            y (:obj:`str`, optional, default is ``None``): The value of the log (data-point).
            timestamp (:obj:`time`, optional, default is ``None``):
                Timestamp to be associated with log entry. Must be Unix time.
                If ``None`` is passed, `time.time() <https://docs.python.org/3.6/library/time.html#time.time>`_
                (Python 3.6 example) is invoked to obtain timestamp.

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                # common case, where log name and data are passed
                neptune.log_text('my_text_data', str(data_item))

                # log_name, x and timestamp are passed
                neptune.log_text(log_name='logging_losses_as_text',
                                 x=str(val_loss),
                                 timestamp=1560430912)

        Note:
            For efficiency, logs are uploaded in batches via a queue.
            Hence, if you log a lot of data, you may experience slight delays in Neptune web application.
        Note:
            Passing ``x`` coordinate as NaN or +/-inf causes this log entry to be ignored.
            Warning is printed to ``stdout``.
        """
        x, y = self._get_valid_x_y(x, y)

        if x is not None and is_nan_or_inf(x):
            x = None

        if not isinstance(y, six.string_types):
            raise InvalidChannelValue(expected_type='str', actual_type=type(y).__name__)

        if x is not None and is_nan_or_inf(x):
            _logger.warning(
                'Invalid metric x-coordinate: %s for channel %s. '
                'Metrics with nan or +/-inf x-coordinates will not be sent to server',
                x,
                log_name)
        else:
            value = ChannelValue(x, dict(text_value=y), timestamp)
            self._channels_values_sender.send(log_name, ChannelType.TEXT.value, value)

    def send_image(self, channel_name, x, y=None, name=None, description=None, timestamp=None):
        """Log image data in Neptune.

        Alias for :meth:`~neptune.experiments.Experiment.log_image`
        """
        return self.log_image(channel_name, x, y, name, description, timestamp)

    def log_image(self, log_name, x, y=None, image_name=None, description=None, timestamp=None):
        """Log image data in Neptune

        | If a log with provided ``log_name`` does not exist, it is created automatically.
        | If log exists (determined by ``log_name``), then new value is appended to it.

        Args:
            log_name (:obj:`str`): The name of log, i.e. `bboxes`, `visualisations`, `sample_images`.
            x (:obj:`double`): Depending, whether ``y`` parameter is passed:

                * ``y`` not passed: The value of the log (data-point). See ``y`` parameter.
                * ``y`` passed: Index of log entry being appended. Must be strictly increasing.

            y (multiple types supported, optional, default is ``None``):

                The value of the log (data-point). Can be one of the following types:

                * :obj:`PIL image`
                  `Pillow docs <https://pillow.readthedocs.io/en/latest/reference/Image.html#image-module>`_
                * :obj:`matplotlib.figure.Figure`
                  `Matplotlib 3.1.1 docs <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.figure.Figure.html>`_
                * :obj:`str` - path to image file
                * 2-dimensional :obj:`numpy.array` - interpreted as grayscale image
                * 3-dimensional :obj:`numpy.array` - behavior depends on last dimension

                    * if last dimension is 1 - interpreted as grayscale image
                    * if last dimension is 3 - interpreted as RGB image
                    * if last dimension is 4 - interpreted as RGBA image

            image_name (:obj:`str`, optional, default is ``None``): Image name
            description (:obj:`str`, optional, default is ``None``): Image description
            timestamp (:obj:`time`, optional, default is ``None``):
                Timestamp to be associated with log entry. Must be Unix time.
                If ``None`` is passed, `time.time() <https://docs.python.org/3.6/library/time.html#time.time>`_
                (Python 3.6 example) is invoked to obtain timestamp.

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                # path to image file
                experiment.log_image('bbox_images', 'pictures/image.png')
                experiment.log_image('bbox_images', x=5, 'pictures/image.png')
                experiment.log_image('bbox_images', 'pictures/image.png', image_name='difficult_case')

                # PIL image
                img = PIL.Image.new('RGB', (60, 30), color = 'red')
                experiment.log_image('fig', img)

                # 2d numpy array
                array = numpy.random.rand(300, 200)*255
                experiment.log_image('fig', array)

                # 3d grayscale array
                array = numpy.random.rand(300, 200, 1)*255
                experiment.log_image('fig', array)

                # 3d RGB array
                array = numpy.random.rand(300, 200, 3)*255
                experiment.log_image('fig', array)

                # 3d RGBA array
                array = numpy.random.rand(300, 200, 4)*255
                experiment.log_image('fig', array)

                # matplotlib figure example 1
                from matplotlib import pyplot
                pyplot.plot([1, 2, 3, 4])
                pyplot.ylabel('some numbers')
                experiment.log_image('plots', plt.gcf())

                # matplotlib figure example 2
                from matplotlib import pyplot
                import numpy

                numpy.random.seed(19680801)
                data = numpy.random.randn(2, 100)

                figure, axs = pyplot.subplots(2, 2, figsize=(5, 5))
                axs[0, 0].hist(data[0])
                axs[1, 0].scatter(data[0], data[1])
                axs[0, 1].plot(data[0], data[1])
                axs[1, 1].hist2d(data[0], data[1])

                experiment.log_image('diagrams', figure)

        Note:
            For efficiency, logs are uploaded in batches via a queue.
            Hence, if you log a lot of data, you may experience slight delays in Neptune web application.
        Note:
            Passing ``x`` coordinate as NaN or +/-inf causes this log entry to be ignored.
            Warning is printed to ``stdout``.
        Warning:
            Only images up to 15MB are supported. Larger files will not be logged to Neptune.
        """
        x, y = self._get_valid_x_y(x, y)

        if x is not None and is_nan_or_inf(x):
            x = None

        image_content = get_image_content(y)
        if len(image_content) > self.IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            _logger.warning('Your image is larger than %dMB. Neptune supports logging images smaller than %dMB. '
                            'Resize or increase compression of this image',
                            self.IMAGE_SIZE_LIMIT_MB,
                            self.IMAGE_SIZE_LIMIT_MB)
            image_content = None

        input_image = dict(
            name=image_name,
            description=description
        )
        if image_content:
            input_image['data'] = base64.b64encode(image_content).decode('utf-8')

        if x is not None and is_nan_or_inf(x):
            _logger.warning(
                'Invalid metric x-coordinate: %s for channel %s. '
                'Metrics with nan or +/-inf x-coordinates will not be sent to server',
                x,
                log_name)
        else:
            value = ChannelValue(x, dict(image_value=input_image), timestamp)
            self._channels_values_sender.send(log_name, ChannelType.IMAGE.value, value)

    def send_artifact(self, artifact, destination=None):
        """Save an artifact (file) in experiment storage.

        Alias for :meth:`~neptune.experiments.Experiment.log_artifact`
        """
        return self.log_artifact(artifact, destination)

    def log_artifact(self, artifact, destination=None):
        """Save an artifact (file) in experiment storage.

        Args:
            artifact (:obj:`str` or :obj:`IO object`):
                A path to the file in local filesystem or IO object. It can be open
                file descriptor or in-memory buffer like `io.StringIO` or `io.BytesIO`.
            destination (:obj:`str`, optional, default is ``None``):
                A destination path.
                If ``None`` is passed, an artifact file name will be used.

        Raises:
            `FileNotFound`: When ``artifact`` file was not found.
            `StorageLimitReached`: When storage limit in the project has been reached.

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                # simple use
                experiment.log_artifact('images/wrong_prediction_1.png')

                # save file in other directory
                experiment.log_artifact('images/wrong_prediction_1.png', 'validation/images/wrong_prediction_1.png')

                # save file under different name
                experiment.log_artifact('images/wrong_prediction_1.png', 'images/my_image_1.png')
        """
        if isinstance(artifact, str):
            if os.path.exists(artifact):
                target_name = os.path.basename(artifact) if destination is None else destination
                upload_entry = UploadEntry(os.path.abspath(artifact), normalize_file_name(target_name))
            else:
                raise FileNotFound(artifact)
        elif hasattr(artifact, 'read'):
            if destination is not None:
                upload_entry = UploadEntry(artifact, normalize_file_name(destination))
            else:
                raise ValueError("destination is required for file streams")
        else:
            raise ValueError("artifact is a local path or an IO object")

        upload_to_storage(upload_entries=[upload_entry],
                          upload_api_fun=self._backend.upload_experiment_output,
                          upload_tar_api_fun=self._backend.extract_experiment_output,
                          experiment=self)

    def delete_artifacts(self, path):
        """Removes an artifact(s) (file/directory) from the experiment storage.

        Args:
            path (:obj:`list` or :obj:`str`): Path or list of paths to remove from the experiment's output

        Raises:
            `FileNotFound`: If a path in experiment artifacts does not exist.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                experiment.delete_artifacts('forest_results.pkl')
                experiment.delete_artifacts(['forest_results.pkl', 'directory'])
                experiment.delete_artifacts('')
        """
        if path is None:
            raise ValueError("path argument must not be None")

        paths = path
        if not isinstance(path, list):
            paths = [path]
        for path in paths:
            if path is None:
                raise ValueError("path argument must not be None")
            normalized_path = os.path.normpath(path)
            if normalized_path.startswith(".."):
                raise ValueError("path to delete must be within project's directory")
            if normalized_path == "." or normalized_path == "/" or not normalized_path:
                raise ValueError("Cannot delete whole artifacts directory")
        try:
            for path in paths:
                self._backend.rm_data(experiment=self, path=path)
        except PathInProjectNotFound:
            raise FileNotFound(path)

    def download_artifact(self, path, destination_dir=None):
        """Download an artifact (file) from the experiment storage.

        Download a file indicated by ``path`` from the experiment artifacts and save it in ``destination_dir``.

        Args:
            path (:obj:`str`): Path to the file to be downloaded.
            destination_dir (:obj:`str`):
                The directory where the file will be downloaded.
                If ``None`` is passed, the file will be downloaded to the current working directory.

        Raises:
            `NotADirectory`: When ``destination_dir`` is not a directory.
            `FileNotFound`: If a path in experiment artifacts does not exist.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                experiment.download_artifact('forest_results.pkl', '/home/user/files/')
        """
        if not destination_dir:
            destination_dir = os.getcwd()

        project_storage_path = "/{exp_id}/output/{file}".format(exp_id=self.id, file=path)
        destination_path = os.path.join(destination_dir, os.path.basename(path))

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        elif not os.path.isdir(destination_dir):
            raise NotADirectory(destination_dir)

        try:
            self._backend.download_data(self._project, project_storage_path, destination_path)
        except PathInProjectNotFound:
            raise FileNotFound(path)

    def download_sources(self, path=None, destination_dir=None):
        """Download a directory or a single file from experiment's sources as a ZIP archive.

        Download a subdirectory (or file) ``path`` from the experiment sources and save it in ``destination_dir``
        as a ZIP archive. The name of an archive will be a name of downloaded directory (or file) with '.zip' extension.

        Args:
            path (:obj:`str`):
                Path of a directory or file in experiment sources to be downloaded.
                If ``None`` is passed, all source files will be downloaded.

            destination_dir (:obj:`str`): The directory where the archive will be downloaded.
                If ``None`` is passed, the archive will be downloaded to the current working directory.

        Raises:
            `NotADirectory`: When ``destination_dir`` is not a directory.
            `FileNotFound`: If a path in experiment sources does not exist.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                # Download all experiment sources to current working directory
                experiment.download_sources()

                # Download a single directory
                experiment.download_sources('src/my-module')

                # Download all experiment sources to user-defined directory
                experiment.download_sources(destination_dir='/tmp/sources/')

                # Download a single directory to user-defined directory
                experiment.download_sources('src/my-module', 'sources/')
        """
        if not path:
            path = ""
        if not destination_dir:
            destination_dir = os.getcwd()

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        elif not os.path.isdir(destination_dir):
            raise NotADirectory(destination_dir)

        download_request = self._backend.prepare_source_download_reuqest(self, path)
        self._download_from_request(download_request, destination_dir, path)

    def download_artifacts(self, path=None, destination_dir=None):
        """Download a directory or a single file from experiment's artifacts as a ZIP archive.

        Download a subdirectory (or file) ``path`` from the experiment artifacts and save it in ``destination_dir``
        as a ZIP archive. The name of an archive will be a name of downloaded directory (or file) with '.zip' extension.

        Args:
            path (:obj:`str`):
                Path of a directory or file in experiment artifacts to be downloaded.
                If ``None`` is passed, all artifacts will be downloaded.

            destination_dir (:obj:`str`): The directory where the archive will be downloaded.
                If ``None`` is passed, the archive will be downloaded to the current working directory.

        Raises:
            `NotADirectory`: When ``destination_dir`` is not a directory.
            `FileNotFound`: If a path in experiment artifacts does not exist.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                # Download all experiment artifacts to current working directory
                experiment.download_artifacts()

                # Download a single directory
                experiment.download_artifacts('data/images')

                # Download all experiment artifacts to user-defined directory
                experiment.download_artifacts(destination_dir='/tmp/artifacts/')

                # Download a single directory to user-defined directory
                experiment.download_artifacts('data/images', 'artifacts/')
        """
        if not path:
            path = ""
        if not destination_dir:
            destination_dir = os.getcwd()

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        elif not os.path.isdir(destination_dir):
            raise NotADirectory(destination_dir)

        download_request = self._backend.prepare_output_download_reuqest(self, path)
        self._download_from_request(download_request, destination_dir, path)

    def _download_from_request(self, download_request, destination_dir, path):
        sleep_time = 1
        max_sleep_time = 16
        while not hasattr(download_request, "downloadUrl"):
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, max_sleep_time)
            download_request = self._backend.get_download_request(download_request.id)

        ssl_verify = True
        if os.getenv("NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"):
            ssl_verify = False

        # We do not use Backend here cause `downloadUrl` can be any url (not only Neptune API endpoint)
        response = requests.get(
            url=download_request.downloadUrl,
            headers={"Accept": "application/zip"},
            stream=True,
            verify=ssl_verify
        )

        with response:
            filename = None
            if 'content-disposition' in response.headers:
                content_disposition = response.headers['content-disposition']
                filenames = re.findall("filename=(.+)", content_disposition)
                if filenames:
                    filename = filenames[0]

            if not filename:
                filename = os.path.basename(path.rstrip("/")) + ".zip"

            destination_path = os.path.join(destination_dir, filename)
            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)

    def reset_log(self, log_name):
        """Resets the log.

        Removes all data from the log and enables it to be reused from scratch.

        Args:
            log_name (:obj:`str`): The name of log to reset.

        Raises:
            `ChannelDoesNotExist`: When the log with name ``log_name`` does not exist on the server.

        Example:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                experiment.reset_log('my_metric')

        Note:
            Check Neptune web application to see that reset charts have no data.
        """
        channel = self._find_channel(log_name, ChannelNamespace.USER)
        if channel is None:
            raise ChannelDoesNotExist(self.id, log_name)
        self._backend.reset_channel(channel.id)

    def get_parameters(self):
        """Retrieve parameters for this experiment.

        Returns:
            :obj:`dict` - dictionary mapping a parameter name to value.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                exp_params = experiment.get_parameters()
        """
        experiment = self._backend.get_experiment(self.internal_id)
        return dict((p.name, self._convert_parameter_value(p.value, p.parameterType)) for p in experiment.parameters)

    def get_properties(self):
        """Retrieve User-defined properties for this experiment.

        Returns:
            :obj:`dict` - dictionary mapping a property key to value.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

            .. code:: python3

                exp_properties = experiment.get_properties()
        """
        experiment = self._backend.get_experiment(self.internal_id)
        return dict((p.key, p.value) for p in experiment.properties)

    def set_property(self, key, value):
        """Set `key-value` pair as an experiment property.

        If property with given ``key`` does not exist, it adds a new one.

        Args:
            key (:obj:`str`): Property key.
            value (:obj:`obj`): New value of a property.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                experiment.set_property('model', 'LightGBM')
                experiment.set_property('magic-number', 7)
        """
        properties = {p.key: p.value for p in self._backend.get_experiment(self.internal_id).properties}
        properties[key] = str(value)
        return self._backend.update_experiment(
            experiment=self,
            properties=properties
        )

    def remove_property(self, key):
        """Removes a property with given key.

        Args:
            key (single :obj:`str`):
                Key of property to remove.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                experiment.remove_property('host')
        """
        properties = {p.key: p.value for p in self._backend.get_experiment(self.internal_id).properties}
        del properties[key]
        return self._backend.update_experiment(
            experiment=self,
            properties=properties
        )

    def get_hardware_utilization(self):
        """Retrieve GPU, CPU and memory utilization data.

        Get hardware utilization metrics for entire experiment as a single
        `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        object. Returned DataFrame has following columns (assuming single GPU with 0 index):

            * `x_ram` - time (in milliseconds) from the experiment start,
            * `y_ram` - memory usage in GB,
            * `x_cpu` - time (in milliseconds) from the experiment start,
            * `y_cpu` - CPU utilization percentage (0-100),
            * `x_gpu_util_0` - time (in milliseconds) from the experiment start,
            * `y_gpu_util_0` - GPU utilization percentage (0-100),
            * `x_gpu_mem_0` - time (in milliseconds) from the experiment start,
            * `y_gpu_mem_0` - GPU memory usage in GB.

        | If more GPUs are available they have their separate columns with appropriate indices (0, 1, 2, ...),
          for example: `x_gpu_util_1`, `y_gpu_util_1`.
        | The returned DataFrame may contain ``NaN`` s if one of the metrics has more values than others.

        Returns:
            :obj:`pandas.DataFrame` - DataFrame containing the hardware utilization metrics.

        Examples:
            The following values denote that after 3 seconds, the experiment used 16.7 GB of RAM

                * `x_ram` = 3000
                * `y_ram` = 16.7

            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                hardware_df = experiment.get_hardware_utilization()
        """
        metrics_csv = self._backend.get_metrics_csv(self)
        try:
            return pd.read_csv(metrics_csv)
        except EmptyDataError:
            return pd.DataFrame()

    def get_numeric_channels_values(self, *channel_names):
        """Retrieve values of specified metrics (numeric logs).

        The returned
        `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        contains 1 additional column `x` along with the requested metrics.

        Args:
            *channel_names (one or more :obj:`str`): comma-separated metric names.

        Returns:
            :obj:`pandas.DataFrame` - DataFrame containing values for the requested metrics.

            | The returned DataFrame may contain ``NaN`` s if one of the metrics has more values than others.

        Example:
            Invoking ``get_numeric_channels_values('loss', 'auc')`` returns DataFrame with columns
            `x`, `loss`, `auc`.

            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                batch_channels = experiment.get_numeric_channels_values('batch-1-loss', 'batch-2-metric')
                epoch_channels = experiment.get_numeric_channels_values('epoch-1-loss', 'epoch-2-metric')

        Note:
            It's good idea to get metrics with common temporal pattern (like iteration or batch/epoch number).
            Thanks to this each row of returned DataFrame has metrics from the same moment in experiment.
            For example, combine epoch metrics to one DataFrame and batch metrics to the other.
        """

        channels_data = {}
        channels_by_name = self.get_channels()
        for channel_name in channel_names:
            channel_id = channels_by_name[channel_name].id
            try:
                channels_data[channel_name] = pd.read_csv(
                    self._backend.get_channel_points_csv(self, channel_id),
                    header=None,
                    names=['x_{}'.format(channel_name), 'y_{}'.format(channel_name)],
                    dtype=float
                )
            except EmptyDataError:
                channels_data[channel_name] = pd.DataFrame(
                    columns=['x_{}'.format(channel_name), 'y_{}'.format(channel_name)],
                    dtype=float
                )

        return align_channels_on_x(pd.concat(channels_data.values(), axis=1, sort=False))

    def _start(self,
               upload_source_entries=None,
               abort_callback=None,
               logger=None,
               upload_stdout=True,
               upload_stderr=True,
               send_hardware_metrics=True,
               run_monitoring_thread=True,
               handle_uncaught_exceptions=True):
        upload_to_storage(upload_entries=upload_source_entries,
                          upload_api_fun=self._backend.upload_experiment_source,
                          upload_tar_api_fun=self._backend.extract_experiment_source,
                          warn_limit=100 * 1024 * 1024,
                          experiment=self)

        self._execution_context.start(
            abort_callback=abort_callback,
            logger=logger,
            upload_stdout=upload_stdout,
            upload_stderr=upload_stderr,
            send_hardware_metrics=send_hardware_metrics,
            run_monitoring_thread=run_monitoring_thread,
            handle_uncaught_exceptions=handle_uncaught_exceptions
        )

    def stop(self, exc_tb=None):
        """Marks experiment as finished (succeeded or failed).

        Args:
            exc_tb (:obj:`str`, optional, default is ``None``): Additional traceback information
                to be stored in experiment details in case of failure (stacktrace, etc).
                If this argument is ``None`` the experiment will be marked as succeeded.
                Otherwise, experiment will be marked as failed.

        Examples:
            Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

            .. code:: python3

                # Marks experiment as succeeded
                experiment.stop()

                # Assuming 'ex' is some exception,
                # it marks experiment as failed with exception info in experiment details.
                experiment.stop(str(ex))
        """

        self._channels_values_sender.join()

        try:
            if exc_tb is None:
                self._backend.mark_succeeded(self)
            else:
                self._backend.mark_failed(self, exc_tb)
        except ExperimentAlreadyFinished:
            pass

        self._execution_context.stop()

        # pylint: disable=protected-access
        self._project._remove_stopped_experiment(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            self.stop()
        else:
            self.stop("\n".join(traceback.format_tb(exc_tb)) + "\n" + repr(exc_val))

    def __str__(self):
        return 'Experiment({})'.format(self.id)

    def __repr__(self):
        return str(self)

    def __eq__(self, o):
        # pylint: disable=protected-access
        return self._id == o._id and self._internal_id == o._internal_id and self._project == o._project

    def __ne__(self, o):
        return not self.__eq__(o)

    @staticmethod
    def _convert_parameter_value(value, parameter_type):
        if parameter_type == 'double':
            return float(value)
        else:
            return value

    @staticmethod
    def _get_valid_x_y(x, y):
        """
        The goal of this function is to allow user to call experiment.log_* with any of:
            - single parameter treated as y value
            - both parameters (named/unnamed)
            - single named y parameter
        If intended X-coordinate is provided, it is validated to be a float value
        """
        if x is None and y is None:
            raise NoChannelValue()

        if x is None and y is not None:
            return None, y

        if x is not None and y is None:
            return None, x

        if x is not None and y is not None:
            if not is_float(x):
                raise InvalidChannelValue(expected_type='float', actual_type=type(x).__name__)
            return x, y

    def _send_channels_values(self, channels_with_values):
        self._backend.send_channels_values(self, channels_with_values)

    def _get_channels(self, channels_names_with_types):
        existing_channels = self.get_channels()
        channels_by_name = {}
        for (channel_name, channel_type) in channels_names_with_types:
            channel = existing_channels.get(channel_name, None)
            if channel is None:
                channel = self._create_channel(channel_name, channel_type)
            channels_by_name[channel.name] = channel
        return channels_by_name

    def _get_channel(self, channel_name, channel_type, channel_namespace=ChannelNamespace.USER):
        channel = self._find_channel(channel_name, channel_namespace)
        if channel is None:
            channel = self._create_channel(channel_name, channel_type, channel_namespace)
        return channel

    def _find_channel(self, channel_name, channel_namespace):
        if channel_namespace == ChannelNamespace.USER:
            return self.get_channels().get(channel_name, None)
        elif channel_namespace == ChannelNamespace.SYSTEM:
            return self._get_system_channels().get(channel_name, None)
        else:
            raise RuntimeError("Unknown channel namespace {}".format(channel_namespace))

    def _create_channel(self, channel_name, channel_type, channel_namespace=ChannelNamespace.USER):
        if channel_namespace == ChannelNamespace.USER:
            return self._backend.create_channel(self, channel_name, channel_type)
        elif channel_namespace == ChannelNamespace.SYSTEM:
            return self._backend.create_system_channel(self, channel_name, channel_type)
        else:
            raise RuntimeError("Unknown channel namespace {}".format(channel_namespace))
