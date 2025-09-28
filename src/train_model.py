# src/train_model.py
"""
Train model script for House Prices project.
Usage:
  Activate venv and run: python src/train_model.py
This will:
 - load data/processed/train_final.csv (or train_clean.csv if final absent)
 - train a Lasso pipeline (log-target)
 - evaluate with 5-fold CV (RMSE)
 - save models/pipeline_v2.joblib and models/pipeline_v2_meta.json
"""
import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.compose import TransformedTargetRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1) load data (prefer train_final.csv if present)
train_final = os.path.join(DATA_PROCESSED, "train_final.csv")
train_clean = os.path.join(DATA_PROCESSED, "train_clean.csv")
if os.path.exists(train_final):
    df = pd.read_csv(train_final)
elif os.path.exists(train_clean):
    df = pd.read_csv(train_clean)
else:
    raise FileNotFoundError("No processed train file found. Run preprocessing first.")

print("Loaded data:", df.shape)

# 2) prepare X, y
if "SalePrice" not in df.columns:
    raise KeyError("train file missing SalePrice column")

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"].values

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Num cols: {len(num_cols)}, Cat cols: {len(cat_cols)}")

# 3) build preprocessor & pipeline (same as notebook)
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
], remainder="drop")

# 4) model: Lasso (chosen champion)
lasso = Lasso(alpha=0.001, random_state=42, max_iter=10000)
pipe = Pipeline([("preprocessor", preprocessor), ("model", lasso)])
ttr = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

# 5) cross-validate
print("Running 5-fold CV (neg_root_mean_squared_error)...")
scores = -cross_val_score(ttr, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
print(f"CV RMSE: mean={scores.mean():.2f}, std={scores.std():.2f}")

# 6) fit final model on full train and save
print("Fitting final model on full training set...")
ttr.fit(X, y)
model_path = os.path.join(MODELS_DIR, "pipeline_v2.joblib")
joblib.dump(ttr, model_path)
print("Saved model:", model_path)

# 7) save metadata
meta = {
    "model": "Lasso (alpha=0.001), TransformedTargetRegressor(log1p)",
    "cv_rmse_mean": float(scores.mean()),
    "cv_rmse_std": float(scores.std()),
    "num_features": len(num_cols) + len(cat_cols),
    "num_numeric": len(num_cols),
    "num_categorical": len(cat_cols),
    "train_shape": df.shape,
    "dropped_columns_note": "Dropped very sparse columns in preprocessing",
    "created_at": datetime.utcnow().isoformat() + "Z"
}
with open(os.path.join(MODELS_DIR, "pipeline_v2_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Saved metadata.")

print("Done.")
