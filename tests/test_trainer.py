import pytest
import torch
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from fiap_tech_challenge_4.config import TrainingConfig, DataStrategyConfig, ModelParams, TechnicalsConfig
from fiap_tech_challenge_4.modeling.trainer import ModelTrainer


@pytest.fixture
def clean_artifacts():
    """Clean up artifacts created during tests."""
    yield
    if Path("artifacts").exists():
        shutil.rmtree("artifacts")


@pytest.fixture
def mock_stock_df():
    import pandas as pd
    import numpy as np
    dates = pd.date_range(start="2023-01-01", periods=50)
    df = pd.DataFrame({
        "open": np.random.rand(50) * 100,
        "high": np.random.rand(50) * 100,
        "low": np.random.rand(50) * 100,
        "close": np.linspace(100, 200, 50),
        "volume": np.random.rand(50) * 1000
    }, index=dates)
    df.index.name = "Date"
    return df.reset_index().rename(columns={"index": "ts"})


@patch("fiap_tech_challenge_4.modeling.trainer.MLFlowLogger")
@patch("fiap_tech_challenge_4.modeling.trainer.mlflow")
@patch("fiap_tech_challenge_4.features.pipeline.fetch_data")
def test_trainer_orchestration(mock_fetch, mock_stock_df, clean_artifacts):
    """
    Verifies that ModelTrainer can:
    1. Build the pipeline
    2. Init the model with correct dimensions
    3. Save artifacts
    """
    mock_fetch.return_value = mock_stock_df

    # Minimal config for speed
    cfg = TrainingConfig(
        experiment_name="test_experiment",
        epochs=1,
        data=DataStrategyConfig(
            ticker="TEST",
            period="10d",
            seq_len=5, 
            batch_size=2,
            train_split=0.5,
            rolling_windows=[5],
            technicals=TechnicalsConfig(sma=[5], macd=None),
        ),
        model={"hidden_dim": 8}  
    )

    trainer = ModelTrainer(cfg)

    # We don't want to wait for actual training in unit tests
    # But we want to ensure the code runs up to saving
    with patch("pytorch_lightning.Trainer.fit") as mock_fit:
        run_id = trainer.train()

        # Assert fit was called
        mock_fit.assert_called_once()

        # Check Artifacts
        # trainer.train() generates a run_id, usually 'local_run' if no mlflow server
        # or a UUID if mlflow is mocked.
        # Let's check if the folder exists.
        artifact_path = Path("artifacts/test_experiment")
        assert artifact_path.exists()

        # Find the run folder (it might be a UUID or 'local_run')
        run_folders = list(artifact_path.iterdir())
        assert len(run_folders) > 0

        saved_folder = run_folders[0]
        assert (saved_folder / "model.pt").exists()
        assert (saved_folder / "feature_scaler.pkl").exists()
        assert (saved_folder / "config.json").exists()
