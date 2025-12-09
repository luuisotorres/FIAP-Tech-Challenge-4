from fastapi import APIRouter, HTTPException, BackgroundTasks, status, Query
from fiap_tech_challenge_4.schemas.requests import (
    TrainingRequest,
    TrainingResponse,
    PredictionRequest,
    PredictionResponse,
    PromotionResponse,
    HealthResponse,
    ModelMetadataResponse,
)
from fiap_tech_challenge_4.services.training import execute_training_job
from fiap_tech_challenge_4.services.inference import predict_next_step
from fiap_tech_challenge_4.services.promotion import promote_model_from_registry
from fiap_tech_challenge_4.core.state import production_model

router = APIRouter()


async def background_training_wrapper(payload: TrainingRequest):
    """Acquires lock and runs training."""
    async with production_model.training_lock:
        try:
            print(f"🔒 Starting training job: {payload.experiment_name}")
            execute_training_job(payload)
        except Exception as e:
            print(f"❌ Background training failed: {e}")


@router.post("/train", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_training(payload: TrainingRequest, background_tasks: BackgroundTasks):
    """Triggers an async training job. Returns 409 if busy."""
    if production_model.training_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Server is busy with another training job."
        )

    background_tasks.add_task(background_training_wrapper, payload)

    return TrainingResponse(
        message="Training accepted. Check logs/MLflow for progress.",
        run_id="pending",
        status="accepted"
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    """Returns the next-day price forecast."""
    try:
        return predict_next_step(payload)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/promote", response_model=PromotionResponse)
def promote(run_id: str = Query(..., description="MLflow Run ID to deploy")):
    """Hot-swaps the active model with a version from MLflow."""
    try:
        return promote_model_from_registry(run_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Liveness probe. Returns 200 if API is up.
    Checks if a model is loaded in memory.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model_loaded=production_model.is_loaded
    )


@router.get("/model", response_model=ModelMetadataResponse)
def get_active_model_info():
    """
    Returns the hyperparameters of the currently active model.
    """
    if not production_model.is_loaded or not production_model.config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model currently loaded."
        )

    cfg = production_model.config

    return ModelMetadataResponse(
        experiment_name=cfg.experiment_name,
        strategy_type=cfg.data.strategy_type,
        seq_len=cfg.data.seq_len,
        model_params=cfg.model.model_dump()
    )
