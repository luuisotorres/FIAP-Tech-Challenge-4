import json
import joblib
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
    """Orchestrates the end-to-end training flow: Data -> Model -> Artifacts."""

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.artifacts_dir = Path("artifacts") / self.cfg.experiment_name

    def train(self) -> str:
        """Executes training and returns the run_id."""

        logger = MLFlowLogger(
            experiment_name=self.cfg.experiment_name,
            log_model=True
        )

        run_id = logger.run_id
        print(f"🚀 Active MLflow Run ID: {run_id}")


        with mlflow.start_run(run_id=run_id):
            # Select Strategy
            if self.cfg.data.strategy_type == "trend":
                strategy = TrendFollowingStrategy(self.cfg.data)
            elif self.cfg.data.strategy_type == "mean_reversion":
                strategy = MeanReversionStrategy(self.cfg.data)
            else:
                raise ValueError(
                    f"Unknown strategy: {self.cfg.data.strategy_type}")

            # Run Pipeline
            pipeline = DataPipeline(strategy, self.cfg.data)
            loaders = pipeline.run()

            # Dynamic Architecture Initialization
            # Detect input_dim from the generated data (Batch, Seq, Features)
            sample_batch, _ = next(iter(loaders["train"]))
            input_dim = sample_batch.shape[2]

            print(f"Detected input dimension: {input_dim}")

            model = StockLSTM(
                input_dim=input_dim,
                **self.cfg.model.model_dump()
            )

            # Setup Lightning
            lit_module = LSTMLightningModule(
                model, learning_rate=self.cfg.learning_rate)

            # Use MLflow run ID if active, else generic
            run = mlflow.active_run()
            run_id = run.info.run_id if run else "local_run"

            logger = MLFlowLogger(
                experiment_name=self.cfg.experiment_name,
                run_id=run_id
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
        """Saves weights, config, and scalers to disk."""
        save_path = self.artifacts_dir / run_id
        save_path.mkdir(parents=True, exist_ok=True)

        # Model Weights
        torch.save(model.state_dict(), save_path / "model.pt")

        # Scalers (CRITICAL for inference)
        joblib.dump(pipeline.feature_scaler, save_path / "feature_scaler.pkl")
        joblib.dump(pipeline.target_scaler, save_path / "target_scaler.pkl")

        # Config
        with open(save_path / "config.json", "w") as f:
            f.write(self.cfg.model_dump_json(indent=2))

        print(f"Artifacts saved to {save_path}")
