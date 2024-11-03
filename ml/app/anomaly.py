from typing import Tuple

import torch
from lightning import LightningModule
from torch.nn import functional as F

from ml.app.models.ood import Autoencoder


class DiseaseOODModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = Autoencoder(in_channels=3, out_channels=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("TL", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        loss = self._loss(batch)
        self.log("VL", loss, prog_bar=True)
        return loss

    def _loss(self, batch):
        images, _ = batch
        outputs = self(images)
        loss = F.mse_loss(outputs, images)
        return loss
