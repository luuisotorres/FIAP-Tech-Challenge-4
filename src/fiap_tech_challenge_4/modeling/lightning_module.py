import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Any
from torchmetrics import MeanAbsoluteError

class LSTMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for LSTM training.

    Handles the training loop, validation loop, metric logging (Loss and MAE),
    and optimizer configuration.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        """
        Initializes the Lightning Module.

        Args:
            model (nn.Module): The underlying PyTorch model (e.g., StockLSTM).
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-3.
        """
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.criterion = nn.MSELoss()
        
        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Executes a single training step.

        Calculates loss (MSE) and Mean Absolute Error (MAE), logs them to the logger,
        and returns the loss for backpropagation.

        Args:
            batch (Tuple): A tuple containing (features, targets).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.
        """
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # Update and log metrics
        self.train_mae(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", self.train_mae, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Executes a single validation step.

        Calculates and logs validation loss and MAE.

        Args:
            batch (Tuple): A tuple containing (features, targets).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.
        """
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # Update and log metrics
        self.val_mae(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", self.val_mae, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self) -> Any:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The Adam optimizer configured with the specified learning rate.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)