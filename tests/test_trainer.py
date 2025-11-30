import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from fiap_tech_challenge_4.config import (
    TrainingConfig,
    DataStrategyConfig,
    TechnicalsConfig
)
from fiap_tech_challenge_4.modeling.trainer import ModelTrainer


@pytest.fixture
def clean_artifacts():
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
def test_trainer_orchestration(mock_fetch, mock_mlflow, mock_logger_cls, mock_stock_df, clean_artifacts):
    # Setup Data Mock
    mock_fetch.return_value = mock_stock_df

    # Setup Logger Mock
    mock_logger_instance = MagicMock()
    type(mock_logger_instance).run_id = PropertyMock(
        return_value="test_run_id")
    mock_logger_cls.return_value = mock_logger_instance

    # Setup Global MLflow Mock (Context Manager AND active_run)
    # Create a "Run Object" mock that has the correct ID
    mock_run_object = MagicMock()
    mock_run_object.info.run_id = "test_run_id"

    # Case A: used in 'with mlflow.start_run() as run:'
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run_object

    # Case B: used via 'mlflow.active_run()'
    mock_mlflow.active_run.return_value = mock_run_object

    # Configuration
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
            technicals=TechnicalsConfig(sma=[5], macd=None)
        ),
        model={"hidden_dim": 8}
    )

    trainer = ModelTrainer(cfg)

    # Execute with mocked Training loop
    with patch("pytorch_lightning.Trainer.fit") as mock_fit:
        run_id = trainer.train()

        # Assertions
        assert run_id == "test_run_id"
        mock_fit.assert_called_once()

        # Verify Artifacts exist in the specific run folder
        expected_path = Path("artifacts/test_experiment/test_run_id")
        assert expected_path.exists()
        assert (expected_path / "model.pt").exists()
        assert (expected_path / "feature_scaler.pkl").exists()
        assert (expected_path / "config.json").exists()
