import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """
    Standard LSTM with an MLP regression head.
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int = 1, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Compression bottleneck to filter noise before final regression
        bottleneck_size = int(hidden_dim * 0.75)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_size, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Select the hidden state of the last time step
        # shape: (batch, hidden_dim)
        last_step = lstm_out[:, -1, :]
        
        # shape: (batch, output_dim)
        return self.head(last_step)