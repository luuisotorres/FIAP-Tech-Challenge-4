import json
import random
import datetime

def generate_mock_json():
    # Configuration matches your defaults
    SEQ_LEN = 60
    MAX_LAG = 50  # SMA-50 is the biggest lag
    BUFFER = 5
    REQUIRED_DAYS = SEQ_LEN + MAX_LAG + BUFFER  # approx 115
    
    # Start with a baseline price
    price = 150.0
    candles = []
    
    print(f"Generating {REQUIRED_DAYS} candles...")

    for _ in range(REQUIRED_DAYS):
        # Random walk logic to make it look like a real stock
        change = random.uniform(-0.02, 0.02)  # +/- 2% move
        price = price * (1 + change)
        
        # Create OHLCV bar around the close price
        open_p = price * (1 + random.uniform(-0.005, 0.005))
        high_p = max(open_p, price) * (1 + random.uniform(0, 0.01))
        low_p = min(open_p, price) * (1 - random.uniform(0, 0.01))
        volume = random.randint(1_000_000, 5_000_000)
        
        candles.append({
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(price, 2),
            "volume": volume
        })

    # Construct the full API payload
    payload = {
        "ticker": "AAPL",
        "candles": candles
    }

    # Print as formatted JSON string
    json_output = json.dumps(payload, indent=2)
    
    # Save to a file for easy copying
    with open("mock_payload.json", "w") as f:
        f.write(json_output)
        
    print(f"✅ Mock payload generated in 'mock_payload.json' ({len(candles)} candles).")
    print("Copy the content of that file and paste it into http://localhost:8000/docs")

if __name__ == "__main__":
    generate_mock_json()