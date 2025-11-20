# TODO
# Implement model training strategies
from abc import ABC, abstractmethod
from fiap_tech_challenge_4.data.pipeline import DataPipeline
from fiap_tech_challenge_4.model.lstm import LSTMFactory
from fiap_tech_challenge_4.model.schemas import TrainingParams


class TrainingStrategy(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_data_pipeline(self) -> DataPipeline:
        pass

    @abstractmethod
    def get_model_factory(self) -> LSTMFactory:
        pass

    @abstractmethod
    def get_training_params(self) -> TrainingParams:
        pass
    