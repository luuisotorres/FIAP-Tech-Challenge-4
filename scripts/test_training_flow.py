# scripts/test_training_flow.py
import os
import mlflow
from dotenv import load_dotenv

# # Ensure we can import from src
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fiap_tech_challenge_4.config import (
    TrainingConfig, 
    DataStrategyConfig, 
    ModelParams,
    TechnicalsConfig
)
from fiap_tech_challenge_4.modeling.trainer import ModelTrainer

def run_smoke_test():
    # Load Environment Variables (DagsHub Credentials)
    load_dotenv()
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("❌ Error: MLFLOW_TRACKING_URI not found in .env")
        return

    print(f"🚀 Connecting to DagsHub MLflow: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    # Define a Real Configuration
    config = TrainingConfig(
        experiment_name="Smoke_Test_V1",
        epochs=5,  # Short run just to test connectivity
        learning_rate=0.001,
        data=DataStrategyConfig(
            ticker="AAPL",
            period="1y",
            strategy_type="trend",
            scaler_type="robust",
            train_split=0.8,
            seq_len=30,
            batch_size=32,
            rolling_windows=[5, 20],
            technicals=TechnicalsConfig(sma=[20, 50])
        ),
        model={
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2
        }
    )

    # Initialize Trainer
    print("⚙️ Initializing ModelTrainer...")
    trainer = ModelTrainer(config)

    # Execute Training
    print("🔥 Starting Training...")
    try:
        run_id = trainer.train()
        print("\n✅ Success! Training completed.")
        print(f"🆔 Run ID: {run_id}")
        print(f"🔗 Check your DagsHub UI: {tracking_uri.replace('.mlflow', '/experiments')}")
    except Exception as e:
        print("\n❌ Training Failed:")
        print(e)

if __name__ == "__main__":
    run_smoke_test()