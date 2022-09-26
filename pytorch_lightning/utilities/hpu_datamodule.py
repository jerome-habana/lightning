# Copyright The PyTorch Lightning team.
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
import logging
import os
import platform
import random
import time
import urllib
from typing import Optional, Tuple
from urllib.error import HTTPError
from warnings import warn

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from pl_examples import _DATASETS_PATH
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from pytorch_lightning.utilities import _HPU_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    import torchvision.datasets

if _HPU_AVAILABLE:
    import habana_frameworks.torch.core as htcore
    import habana_dataloader

class HPUDataModule(LightningDataModule):

    name = "hpu-dataset"

    def __init__(
        self,
        dataset_train = None,
        dataset_test = None,
        data_dir: str = _DATASETS_PATH,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        transforms = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if num_workers and platform.system() == "Windows":
            # see: https://stackoverflow.com/a/59680818
            warn(
                f"You have requested num_workers={num_workers} on Windows,"
                " but currently recommended is 0, so we set it for you"
            )
            num_workers = 0

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.transforms = transforms

        self.data_loader_type = torch.utils.data.DataLoader

    def setup(self, stage: Optional[str] = None):
        if isinstance(self.dataset_train, torchvision.datasets.ImageFolder):
            self.data_loader_type = habana_dataloader.HabanaDataLoader
            self.num_workers = 8
        else:
            #raise ValueError("HabanaDataLoader supports only ImageFolder as dataset")
            print("HabanaDataLoader supports only ImageFolder as dataset")
        
        # check supported transforms
        if self.transforms != None:
            for t in self.transforms:
                if isinstance(t, transform_lib.RandomResizedCrop) or \
                    isinstance(t, transform_lib.CenterCrop) or \
                    isinstance(t, transform_lib.Resize) or \
                    isinstance(t, transform_lib.RandomHorizontalFlip) or \
                    isinstance(t, transform_lib.ToTensor) or \
                    isinstance(t, transform_lib.Normalize):
                    
                    continue
                else:
                    raise ValueError("Unsupported transform: " + str(type(t)))
                
    def train_dataloader(self):
        """train set removes a subset to use for validation."""
        shuffle = True
        if _HPU_AVAILABLE:
            shuffle=False
        loader = self.data_loader_type(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """val set uses a subset of the training set for validation."""
        shuffle = True
        if _HPU_AVAILABLE:
            shuffle=False
        loader = self.data_loader_type(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        """test set uses the test split."""
        shuffle = True
        if _HPU_AVAILABLE:
            shuffle=False
        loader = self.data_loader_type(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
