import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Any


class LSTMLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for LSTM training and logging."""

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("train_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
