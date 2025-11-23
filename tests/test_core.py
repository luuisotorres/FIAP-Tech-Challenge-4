import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from fiap_tech_challenge_4.core.state import ModelArtifacts


@pytest.fixture
def mock_artifacts_dir():
    return Path("dummy/artifacts")


@patch("fiap_tech_challenge_4.core.state.StockLSTM")
@patch("fiap_tech_challenge_4.core.state.torch.load")
@patch("fiap_tech_challenge_4.core.state.joblib.load")
@patch("builtins.open", new_callable=mock_open, read_data='{"experiment_name": "test", "data": {"ticker": "AAPL"}}')
def test_load_production_model(mock_file, mock_joblib, mock_torch, mock_artifacts_dir):
    # Setup mocks
    mock_joblib.return_value = MagicMock(n_features_in_=5)  # Mock Scaler
    mock_torch.return_value = {}  # Mock state_dict

    # Mock Path.exists to always return True
    with patch("pathlib.Path.exists", return_value=True):
        artifacts = ModelArtifacts()
        artifacts.load_production_model(mock_artifacts_dir)

        assert artifacts.is_loaded is True
        assert artifacts.model is not None
        assert artifacts.feature_scaler is not None
        # Verify it tried to load the right files
        assert mock_joblib.call_count == 2  # Feature and Target scalers
        mock_torch.assert_called_once()


def test_load_fails_if_dir_missing():
    artifacts = ModelArtifacts()
    with pytest.raises(FileNotFoundError):
        artifacts.load_production_model(Path("non_existent_path"))
