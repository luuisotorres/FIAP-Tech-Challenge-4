from pydantic import BaseModel


class LSTMParams(BaseModel):
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool


class TrainingParams(BaseModel):
    period: str
    pass
