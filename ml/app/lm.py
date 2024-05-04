from typing import Tuple

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


class ClassificationModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("TL", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> Tuple[str, torch.Tensor]:
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = self._accuracy(labels, outputs)
        self.log("VL", loss, prog_bar=True, logger=True)
        self.log("VA", acc, prog_bar=True, logger=True)
        return {"VL": loss, "VA": acc}

    def _accuracy(self, labels, outputs):
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)
        return acc

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("TL", loss, logger=True)
