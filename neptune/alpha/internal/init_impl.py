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
import logging
import os
from datetime import datetime

from pathlib import Path
from platform import node as get_hostname
from typing import Optional, List

import click
from neptune.alpha.internal.operation import Operation

from neptune.alpha.internal.utils import verify_type, verify_collection_type

from neptune.alpha.constants import NEPTUNE_EXPERIMENT_DIRECTORY, OFFLINE_DIRECTORY, \
    ASYNC_DIRECTORY
from neptune.alpha.envs import PROJECT_ENV_NAME
from neptune.alpha.exceptions import MissingProject
from neptune.alpha.internal.backgroud_job_list import BackgroundJobList
from neptune.alpha.internal.hardware.hardware_metric_reporting_job import HardwareMetricReportingJob
from neptune.alpha.internal.operation_processors.async_operation_processor import AsyncOperationProcessor
from neptune.alpha.internal.backends.hosted_neptune_backend import HostedNeptuneBackend
from neptune.alpha.internal.backends.neptune_backend_mock import NeptuneBackendMock
from neptune.alpha.internal.containers.disk_queue import DiskQueue
from neptune.alpha.internal.credentials import Credentials
from neptune.alpha.internal.operation_processors.sync_operation_processor import SyncOperationProcessor
from neptune.alpha.internal.operation_processors.offline_operation_processor import OfflineOperationProcessor
from neptune.alpha.internal.streams.std_capture_background_job import StdoutCaptureBackgroundJob, \
    StderrCaptureBackgroundJob
from neptune.alpha.internal.utils.git import get_git_info, discover_git_repo_location
from neptune.alpha.internal.utils.ping_background_job import PingBackgroundJob
from neptune.alpha.version import version as parsed_version
from neptune.alpha.experiment import Experiment


__version__ = str(parsed_version)

_logger = logging.getLogger(__name__)


def init(
        project: Optional[str] = None,
        experiment: Optional[str] = None,
        connection_mode: str = "async",
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: List[str] = None,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
        capture_hardware_metrics: bool = True,
        flush_period: float = 5) -> Experiment:
    verify_type("project", project, (str, type(None)))
    verify_type("experiment", experiment, (str, type(None)))
    verify_type("connection_mode", connection_mode, str)
    verify_type("name", name, (str, type(None)))
    verify_type("description", description, (str, type(None)))
    verify_type("capture_stdout", capture_stdout, bool)
    verify_type("capture_stderr", capture_stderr, bool)
    verify_type("capture_hardware_metrics", capture_hardware_metrics, bool)
    verify_type("flush_period", flush_period, (int, float))
    if tags:
        verify_collection_type("tags", tags, str)

    name = "Untitled" if experiment is None and name is None else name
    description = "" if experiment is None and description is None else description
    hostname = get_hostname() if experiment is None else None

    if not project:
        project = os.getenv(PROJECT_ENV_NAME)
    if connection_mode == 'offline':
        project = 'offline-project-placeholder'
    if not project:
        raise MissingProject()

    if connection_mode == "async":
        # TODO Initialize backend in async thread
        backend = HostedNeptuneBackend(Credentials())
    elif connection_mode == "sync":
        backend = HostedNeptuneBackend(Credentials())
    elif connection_mode == "debug":
        backend = NeptuneBackendMock()
    elif connection_mode == "offline":
        backend = NeptuneBackendMock()
    else:
        raise ValueError('connection_mode should be one of ["async", "sync", "offline", "debug"]')

    project_obj = backend.get_project(project)
    if experiment:
        exp = backend.get_experiment(project + '/' + experiment)
    else:
        git_ref = get_git_info(discover_git_repo_location())
        exp = backend.create_experiment(project_obj.uuid, git_ref)

    if connection_mode == "async":
        experiment_path = "{}/{}/{}".format(NEPTUNE_EXPERIMENT_DIRECTORY, ASYNC_DIRECTORY, exp.uuid)
        try:
            execution_id = len(os.listdir(experiment_path))
        except FileNotFoundError:
            execution_id = 0
        execution_path = "{}/exec-{}-{}".format(experiment_path, execution_id, datetime.now())
        operation_processor = AsyncOperationProcessor(
            exp.uuid,
            DiskQueue(Path(execution_path), lambda x: x.to_dict(), Operation.from_dict),
            backend,
            sleep_time=flush_period)
    elif connection_mode == "sync":
        operation_processor = SyncOperationProcessor(exp.uuid, backend)
    elif connection_mode == "debug":
        operation_processor = SyncOperationProcessor(exp.uuid, backend)
    elif connection_mode == "offline":
        # Experiment was returned by mocked backend and has some random UUID.
        experiment_path = "{}/{}/{}".format(NEPTUNE_EXPERIMENT_DIRECTORY, OFFLINE_DIRECTORY, exp.uuid)
        storage_queue = DiskQueue(Path(experiment_path), lambda x: x.to_dict(), Operation.from_dict)
        operation_processor = OfflineOperationProcessor(storage_queue)
    else:
        raise ValueError('connection_mode should be on of ["async", "sync", "offline", "debug"]')

    background_jobs = []
    if capture_hardware_metrics:
        if HardwareMetricReportingJob.requirements_installed():
            background_jobs.append(HardwareMetricReportingJob())
        else:
            _logger.warning('psutil is not installed. Hardware metrics will not be collected.')
            background_jobs.append(PingBackgroundJob())
    else:
        background_jobs.append(PingBackgroundJob())
    if capture_stdout:
        background_jobs.append(StdoutCaptureBackgroundJob())
    if capture_stderr:
        background_jobs.append(StderrCaptureBackgroundJob())

    _experiment = Experiment(exp.uuid, backend, operation_processor, BackgroundJobList(background_jobs))
    _experiment.sync(wait=False)

    if name is not None:
        _experiment["sys/name"] = name
    if description is not None:
        _experiment["sys/description"] = description
    if hostname is not None:
        _experiment["sys/hostname"] = hostname
    if tags is not None:
        _experiment["sys/tags"] = tags

    _experiment.start()

    click.echo("{base_url}/{workspace}/{project}/e/{exp_id}".format(
        base_url=backend.get_display_address(),
        workspace=exp.workspace,
        project=exp.project_name,
        exp_id=exp.short_id
    ))

    return _experiment