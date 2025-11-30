import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from typing import Tuple, Any
from torchmetrics import MeanAbsoluteError, MeanSquaredError

class LSTMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper with robust metric logging and visualization.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.criterion = nn.MSELoss()
        
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        
        self.val_preds = []
        self.val_targets = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # Update and log MAE
        self.train_mae(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", self.train_mae, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # Update MAE
        self.val_mae(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", self.val_mae, prog_bar=True, on_step=False, on_epoch=True)
        
        # Store first batch for plotting
        if batch_idx == 0:
            self.val_preds = preds.detach().cpu()
            self.val_targets = y.detach().cpu()
            
        return loss

    def on_validation_epoch_end(self):
        """
        Generates a plot of Prediction vs Actuals and logs it to MLflow.
        This runs at the end of every validation epoch.
        """
        # Skip plotting sanity checks (when logger might not be ready)
        if self.current_epoch == 0 and self.global_step == 0:
            return

        # Create Figure
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot first 50 points to keep it readable
        limit = 50
        ax.plot(self.val_targets[:limit], label="Actual Price (Scaled)", color="blue", marker="o", markersize=3)
        ax.plot(self.val_preds[:limit], label="Predicted (Scaled)", color="orange", linestyle="--", marker="x", markersize=3)
        ax.set_title(f"Validation Forecast (Epoch {self.current_epoch})")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Normalized Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig) # Clean up memory

        # Log to MLflow
        if self.logger:
            # MLflow logger exposes .experiment (Client) and .run_id
            try:
                self.logger.experiment.log_artifact(
                    self.logger.run_id, 
                    local_path=None, # Not used when passing a file-like object? 
                )
                temp_path = "val_plot.png"
                with open(temp_path, "wb") as f:
                    f.write(buf.getbuffer())
                
                self.logger.experiment.log_artifact(
                    run_id=self.logger.run_id,
                    local_path=temp_path,
                    artifact_path="plots"
                )
            except Exception as e:
                print(f"⚠️ Failed to log plot: {e}")

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)