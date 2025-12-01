import joblib
from pathlib import Path
import pandas as pd

proj_root = Path(__file__).resolve().parent.parent
model_path = proj_root / "models" / "house_price_model.pkl"

model = joblib.load(model_path)

def predict_price(input_data: dict):
    
    df = pd.DataFrame([input_data])
    price = model.predict(df)[0]
    return int(price)
