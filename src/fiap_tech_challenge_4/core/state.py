import json
import torch
import joblib
from pathlib import Path
from typing import Optional

from fiap_tech_challenge_4.config import TrainingConfig
from fiap_tech_challenge_4.modeling.lstm import StockLSTM


class ModelArtifacts:
    """Singleton to hold the active production model in memory."""

    def __init__(self):
        self.model: Optional[StockLSTM] = None
        self.config: Optional[TrainingConfig] = None
        self.feature_scaler = None
        self.target_scaler = None
        self.is_loaded = False

    def load_production_model(self, artifact_dir: Path):
        """Loads weights, config, and scalers from disk into memory."""
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Production artifacts not found at {artifact_dir}")

        print(f"Loading production model from {artifact_dir}...")

        # Load Config
        with open(artifact_dir / "config.json", "r") as f:
            config_dict = json.load(f)
            self.config = TrainingConfig(**config_dict)

        # Load Scalers
        self.feature_scaler = joblib.load(artifact_dir / "feature_scaler.pkl")
        self.target_scaler = joblib.load(artifact_dir / "target_scaler.pkl")

        # Initialize Model Architecture
        input_dim = self.feature_scaler.n_features_in_

        self.model = StockLSTM(
            input_dim=input_dim,
            **self.config.model.model_dump()
        )

        # Load Weights
        state_dict = torch.load(artifact_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to inference mode (disables dropout)

        self.is_loaded = True
        print("✅ Production model loaded successfully.")


# Global Instance
production_model = ModelArtifacts()
