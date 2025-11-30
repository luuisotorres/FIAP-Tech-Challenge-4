import json
import joblib
import os
import torch
import mlflow
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import MLFlowLogger
from typing import Dict, Any

from fiap_tech_challenge_4.config import TrainingConfig
from fiap_tech_challenge_4.features.strategies import TrendFollowingStrategy, MeanReversionStrategy
from fiap_tech_challenge_4.features.pipeline import DataPipeline
from fiap_tech_challenge_4.modeling.lstm import StockLSTM
from fiap_tech_challenge_4.modeling.lightning_module import LSTMLightningModule


class ModelTrainer:
    """
    Orchestrates the training lifecycle: Strategy Selection -> Pipeline -> Training -> Artifacts.
    """

    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: The master training configuration.
        """
        self.cfg = config
        self.artifacts_dir = Path("artifacts") / self.cfg.experiment_name
        
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        if env_uri:
            self.tracking_uri = env_uri
        else:
            # Force local file store to avoid root pollution
            self.tracking_uri = "file:./mlruns"
            print("⚠️ Warning: MLFLOW_TRACKING_URI not set. Logging locally to ./mlruns")

    def train(self) -> str:
        """
        Executes the full training job.

        Returns:
            str: The MLflow Run ID generated for this job.
        """
        # Ensure global client points to the right server
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            
        mlflow.set_experiment(self.cfg.experiment_name)

        # Start the Run FIRST to establish the Run ID
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"🚀 Active MLflow Run ID: {run_id}")
            print(f"📡 Logging to: {self.tracking_uri}")

            # Strategy Selection
            if self.cfg.data.strategy_type == "trend":
                strategy = TrendFollowingStrategy(self.cfg.data)
            elif self.cfg.data.strategy_type == "mean_reversion":
                strategy = MeanReversionStrategy(self.cfg.data)
            else:
                raise ValueError(f"Unknown strategy: {self.cfg.data.strategy_type}")

            # Pipeline Execution
            pipeline = DataPipeline(strategy, self.cfg.data)
            loaders = pipeline.run()
            
            # Dynamic Model Initialization
            sample_batch, _ = next(iter(loaders["train"]))
            input_dim = sample_batch.shape[2]
            print(f"Detected input dimension: {input_dim}")

            model = StockLSTM(
                input_dim=input_dim,
                **self.cfg.model.model_dump()
            )

            lit_module = LSTMLightningModule(model, learning_rate=self.cfg.learning_rate)

            # Attach Logger to the EXISTING Run
            logger = MLFlowLogger(
                experiment_name=self.cfg.experiment_name,
                run_id=run_id,
                tracking_uri=self.tracking_uri,
                log_model=True,
                save_dir="./mlruns"  # Ensure fallback logs go here, not root
            )

            trainer = pl.Trainer(
                max_epochs=self.cfg.epochs,
                logger=logger,
                accelerator="auto",
                devices=1,
                enable_progress_bar=True,
                log_every_n_steps=1
            )

            # Train
            trainer.fit(lit_module, loaders["train"], loaders["val"])

            # Save Artifacts
            self._save_artifacts(pipeline, model, run_id)
            
            return run_id

    def _save_artifacts(self, pipeline: DataPipeline, model: torch.nn.Module, run_id: str):
        """Persists model weights, scalers, and configuration to disk."""
        save_path = self.artifacts_dir / run_id
        save_path.mkdir(parents=True, exist_ok=True)

        # Weights
        torch.save(model.state_dict(), save_path / "model.pt")

        # Scalers (Required for inference)
        joblib.dump(pipeline.feature_scaler, save_path / "feature_scaler.pkl")
        joblib.dump(pipeline.target_scaler, save_path / "target_scaler.pkl")

        # Configuration
        with open(save_path / "config.json", "w") as f:
            f.write(self.cfg.model_dump_json(indent=2))

        print(f"Artifacts saved to {save_path}")