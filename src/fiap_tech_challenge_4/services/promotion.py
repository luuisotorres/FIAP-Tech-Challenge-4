import mlflow
import shutil
from pathlib import Path
from fiap_tech_challenge_4.core.state import production_model


PRODUCTION_PATH = Path("artifacts/production")


def promote_model_from_registry(run_id: str):
    """Downloads artifacts from MLflow and hot-swaps the production model."""
    print(f"🔄 Promoting Run ID: {run_id}...")

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    if not run:
        raise ValueError(f"Run ID {run_id} not found in MLflow.")

    # Refresh Production Folder
    if PRODUCTION_PATH.exists():
        shutil.rmtree(PRODUCTION_PATH)
    PRODUCTION_PATH.mkdir(parents=True, exist_ok=True)

    # Download
    for artifact in ["model.pt", "config.json", "feature_scaler.pkl", "target_scaler.pkl"]:
        client.download_artifacts(run_id, artifact, str(PRODUCTION_PATH))

    # Reload Singleton
    production_model.load_production_model(PRODUCTION_PATH)

    return {
        "status": "success",
        "message": f"Model from Run {run_id} is now live.",
        "current_run_id": run_id
    }
