# Copyright The Lightning AI team.
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
from contextlib import contextmanager
from datetime import timedelta
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from torch import Tensor
import torch.distributed
from torch.nn import Module

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.accelerators.hpu import _HPU_AVAILABLE
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.imports import _TORCH_LESSER_EQUAL_1_13_1
from lightning.fabric.plugins.precision.hpu import HPUPrecision
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.utilities.types import Optimizable

if _HPU_AVAILABLE:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed.hccl  # noqa: F401

log = logging.getLogger(__name__)


class HPUParallelStrategy(DDPStrategy):
    """Strategy for distributed training on multiple HPU devices."""

    strategy_name = "parallel_hpu"

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[HPUPrecision] = None,
        process_group_backend: Optional[str] = "hccl",
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        **kwargs: Any,
    ) -> None:

        if not _HPU_AVAILABLE:
            raise ValueError("`HPUParallelStrategy` requires HPU devices to run")

        self._process_group_backend: Optional[str] = "hccl"
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method=start_method,
            **kwargs,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = TorchCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    def setup_environment(self) -> None:

        os.environ["ID"] = str(self.local_rank)
        if self._process_group_backend == "hccl":
            # this env is used in overrides to check the backend initiated
            os.environ["HCCL_DISTRIBUTED_BACKEND"] = str(1)
        super().setup_environment()

    def determine_ddp_device_ids(self) -> None:
        return None

    # def broadcast(self, obj: object, src: int = 0) -> object:  # type: ignore
    #     obj = [obj]
    #     if self.global_rank != src:
    #         obj = [None]

    #     broadcast_object_list(obj, src, group=_group.WORLD)
    #     return obj[0]

    def backward(self, tensor: Tensor, module: Optional[Module], *args: Any, **kwargs: Any) -> None:
        super().backward(tensor=Tensor, module=module, args=args, kwargs=kwargs)
        if _TORCH_LESSER_EQUAL_1_13_1:
            # Break lazy accumulation of graph after fwd+bwd
            htcore.mark_step()

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        optimizer_output = super().optimizer_step(optimizer=optimizer, kwargs=kwargs)
        if _TORCH_LESSER_EQUAL_1_13_1:
            # Break lazy accumulation of graph after optimizer
            htcore.mark_step()
        return optimizer_output

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def teardown(self) -> None:
        super().teardown()
        # Was set to local rank
        os.environ.pop("ID", None)
        os.environ.pop("HCCL_DISTRIBUTED_BACKEND", None)
