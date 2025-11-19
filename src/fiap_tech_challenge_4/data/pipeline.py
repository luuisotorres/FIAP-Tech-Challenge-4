# TODO
# Implement different data processing strategies

from abc import ABC, abstractmethod


class DataStrategy(ABC):
    
    @abstractmethod
    def process():
        pass

    @staticmethod
    def load_data():
        pass

    @staticmethod
    def create_sequences():
        pass


class DataPipeline:
    def __init__(self, strategy: DataStrategy):
        pass

    def run(self):
        pass



