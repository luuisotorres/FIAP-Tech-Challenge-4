from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
from dotenv import load_dotenv

from fiap_tech_challenge_4.api.routes import router
from fiap_tech_challenge_4.core.state import production_model


load_dotenv()
ARTIFACTS_DIR = Path("artifacts")

description = """
# LSTM Stock Prediction API

## Overview
This API provides stock price predictions using an LSTM neural network.
It supports **Training**, **Inference**, and **Model Promotion** via MLflow.

## Features
* **Train:** Trigger async training jobs with custom strategies.
* **Predict:** Get next-day price forecasts.
* **Promote:** Hot-swap the live model with a specific MLflow run ID.

This project was developed for **Tech Challenge 4 - Deep Learning and AI Module**,  
and implemented by the team:

- Izabelly de Oliveira Menezes  
- Larissa Diniz da Silva  
- Luis Fernando Torres  
- Rafael Dos Santos Callegari  
- Renato Massamitsu Zama Inomata  
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: Load model. Shutdown: Cleanup."""
    print("🚀 API Starting...")
    prod_path = ARTIFACTS_DIR / "production"

    if prod_path.exists():
        try:
            production_model.load_production_model(prod_path)
        except Exception as e:
            print(f"⚠️ Failed to load production model: {e}")
    else:
        print("No production model found. Training required.")

    yield
    print("🛑 API Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stock Forecaster",
        description=description,
        version="1.0.0",
        lifespan=lifespan,
        contact={"name": "Tech Challenge 4 Group", "email": "group@fiap.com"}
    )
    app.include_router(router, prefix="/v1")
    return app


app = create_app()
