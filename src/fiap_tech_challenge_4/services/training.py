from fiap_tech_challenge_4.schemas.requests import TrainingRequest
from fiap_tech_challenge_4.config import TrainingConfig
from fiap_tech_challenge_4.modeling.trainer import ModelTrainer


def execute_training_job(request: TrainingRequest) -> str:
    """
    Parses the API request, builds the config, and runs the trainer.
    Returns the run_id.
    """
    # Convert API Request -> Internal Config
    # Since request fields match Config fields, we can unpack
    config = TrainingConfig(
        experiment_name=request.experiment_name,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        data=request.data,
        model=request.model
    )

    trainer = ModelTrainer(config)
    run_id = trainer.train()

    return run_id
