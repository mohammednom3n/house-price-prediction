import sys
from pathlib import Path
import numpy as np
import pandas as pd

proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

# 1) LOAD DATA

RAW_DATA = proj_root / "data" / "raw"

train_path = RAW_DATA / "train.csv"

df = pd.read_csv(train_path)

# 2) Pre-Pipeline Manual Cleaning

df.drop(columns = ["PoolQC","MiscFeature","Alley", "Fence", "MasVnrType"], inplace = True)

# Fill null values
missing = df.isnull().sum()
missing_cols = missing[missing>0].sort_values(ascending = False).index.tolist()
missing_cols

for col in missing_cols:
    df[col] = df[col].fillna("None")
cat_cols = df.select_dtypes(include = ["O"]).columns
df[cat_cols] = df[cat_cols].astype(str)

# outliers removal
outliers = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
df = df[~outliers]

# 3) Define Target and Features
X = df.drop(columns = ["SalePrice"])
y = df["SalePrice"]

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)

cat_features = X_train.select_dtypes(include = ["O"]).columns
num_features = X_train.select_dtypes(include = ["int64", "float64"]).columns


# For production, we need to pick top 15 features and retrain the model on these important features
top_features = [
    "OverallQual",
    "GrLivArea",
    "1stFlrSF",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "LotArea",
    "GarageCars",
    "TotRmsAbvGrd",
    "2ndFlrSF",
    "YearBuilt",
    "GarageArea",
    "FullBath",
    "OverallCond",
    "YearRemodAdd",
    "MSSubClass"
]

# Reduced columns
X_train_red = X_train[top_features]
X_test_red  = X_test[top_features]

# 4) Building a pipeline:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor

final_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", CatBoostRegressor(
        iterations=700,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )),
])


# 5) Train model
final_pipeline.fit(X_train_red, y_train)

# 6) Evaluate model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = final_pipeline.predict(X_test_red)

mae  = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

mape = (abs((y_test - y_pred) / y_test)).mean() * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("MAPE %:", mape)

# 7) Save model

import joblib

model_dir = proj_root / "models"
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "house_price_model.pkl"
joblib.dump(final_pipeline, model_path)

print(f"Model saved to: {model_path}")