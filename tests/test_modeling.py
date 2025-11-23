import torch
import pytest
from fiap_tech_challenge_4.modeling.lstm import StockLSTM
from fiap_tech_challenge_4.modeling.lightning_module import LSTMLightningModule


def test_lstm_shapes():
    # Setup: Batch=32, Seq=10, Features=5
    input_dim = 5
    hidden_dim = 16
    batch_size = 32
    seq_len = 10

    model = StockLSTM(input_dim, hidden_dim, output_dim=1)

    # Dummy input (Batch, Seq, Features)
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    output = model(x)

    # Expect (Batch, 1)
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()


def test_lightning_module_step():
    # Setup
    model = StockLSTM(input_dim=5, hidden_dim=16)
    lit_module = LSTMLightningModule(model, learning_rate=0.01)

    # Create a fake batch
    x = torch.randn(10, 20, 5)  # Batch 10, Seq 20, Feat 5
    y = torch.randn(10, 1)     # Targets

    # Simulate training step
    loss = lit_module.training_step((x, y), batch_idx=0)

    # Loss should be a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
