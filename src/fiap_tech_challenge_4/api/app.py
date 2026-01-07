from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

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
* **Health:** Liveness probe.
* **Model:** Returns the hyperparameters of the currently active model.

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
    )
    app.include_router(router, prefix="/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import HTMLResponse
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Forecaster API</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    color: #eee;
                    min-height: 100vh;
                    margin: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .container {
                    text-align: center;
                    padding: 40px;
                    background: rgba(255,255,255,0.05);
                    border-radius: 16px;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }
                h1 { color: #00d4ff; margin-bottom: 10px; }
                p { color: #aaa; margin-bottom: 30px; }
                .btn {
                    display: inline-block;
                    padding: 12px 24px;
                    margin: 10px;
                    background: linear-gradient(90deg, #00d4ff, #0099ff);
                    color: #fff;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: bold;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                .btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 20px rgba(0,212,255,0.4);
                }
                .btn-grafana {
                    background: linear-gradient(90deg, #00d4ff, #0099ff);
                }
                .btn-grafana:hover {
                    box-shadow: 0 4px 20px rgba(0,212,255,0.4);
                }
                .team { margin-top: 40px; font-size: 12px; color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Stock Forecaster API</h1>
                <p>LSTM-powered stock price predictions with MLOps observability</p>
                <a href="/docs" class="btn">🔍 Swagger UI</a>
                <a href="/redoc" class="btn">📖 ReDoc</a>
                <a href="/v1/health" class="btn">💚 Health Check</a>
                <a href="http://localhost:3000/d/fastapi-observability/" class="btn btn-grafana">📊 Grafana Dashboard</a>
                <div class="team">
                    FIAP Tech Challenge 4 | ML Engineering Postgraduate Program
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    Instrumentator().instrument(app).expose(app)
    
    return app


app = create_app()
