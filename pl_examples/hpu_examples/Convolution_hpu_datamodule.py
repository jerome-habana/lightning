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

import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.utilities import _HPU_AVAILABLE

from pytorch_lightning.callbacks import Callback

from pytorch_lightning.utilities.hpu_datamodule import HPUDataModule


class ConvolutionOnHPU(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.l1 = torch.nn.Linear(28 * 28 * 16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

class DataLayoutPlugin(Callback):
    def __init__(self, model):
        self.model = model

        #permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
        #and permute from RSCK to KCRS is used for checkpoint saving
        def permute_params(model, to_filters_last):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if(param.ndim == 4):
                        if to_filters_last:
                            param.data = param.data.permute((2, 3, 1, 0))
                        else:
                            param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS
        
        for layer in model.children():
            if isinstance(layer, nn.Conv1d):
                print('Found conv1d layer........')
            
            elif isinstance(layer, nn.Conv2d):
                print('Found conv2d layer........')
                if _HPU_AVAILABLE:
                    # Gaudi HW performs convolution operations with filter (weights) in filters last format
                    permute_params(self.model, True)
            
            elif isinstance(layer, nn.Conv3d):
                print('$  Found conv3d layer........')


# Init our model
model = ConvolutionOnHPU()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
val_ds = MNIST(os.getcwd(), train=False, transform=transforms.ToTensor())

data_module = HPUDataModule(train_ds, val_ds)

train_loader = data_module.train_dataloader()
val_loader = data_module.test_dataloader()

# Initialize a trainer
trainer = pl.Trainer(devices=1, accelerator="hpu", max_epochs=3, precision=32, callbacks=[DataLayoutPlugin(model)])

# Train the model ⚡
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model, val_loader)
trainer.validate(model, dataloaders=val_loader)
