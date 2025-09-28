# src/preprocess.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict

# Domain mappings for ordinal features (common in Ames)
QUALITY_MAP = {
    "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0
}
CONDITION_MAP = {
    "Excellent": 5, "Good": 4, "Typical": 3, "Fair": 2, "Poor": 1
}

def drop_sparse_columns(df_train: pd.DataFrame, df_test: pd.DataFrame, threshold: float = 0.5):
    frac = df_train.isnull().mean()
    to_drop = frac[frac > threshold].index.tolist()
    df_train = df_train.drop(columns=to_drop, errors="ignore")
    df_test  = df_test.drop(columns=[c for c in to_drop if c in df_test.columns], errors="ignore")
    return df_train, df_test, to_drop

def impute_by_rules(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Keep MasVnrType, impute MasVnrArea
    if "MasVnrType" in df_train.columns:
        df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
        df_test["MasVnrType"]  = df_test["MasVnrType"].fillna("None")
    if "MasVnrArea" in df_train.columns:
        df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)
        df_test["MasVnrArea"]  = df_test["MasVnrArea"].fillna(0)

    # Categorical absence -> 'None' (auto-detect by keywords)
    none_keywords = ["Garage","Bsmt","Fireplace","Pool","Alley","Fence","Misc"]
    cat_with_na = [c for c in df_train.select_dtypes(include=["object"]).columns if df_train[c].isnull().any()]
    none_cols = [c for c in cat_with_na if any(k in c for k in none_keywords)]
    for c in none_cols:
        df_train[c] = df_train[c].fillna("None")
        df_test[c]  = df_test[c].fillna("None")

    # Numeric absence -> 0 (by keyword)
    num_with_na = [c for c in df_train.select_dtypes(include=[np.number]).columns if df_train[c].isnull().any()]
    zero_keywords = ["Area","SF","Cars","Bath","YrBlt"]
    zero_cols = [c for c in num_with_na if any(k in c for k in zero_keywords)]
    for c in zero_cols:
        df_train[c] = df_train[c].fillna(0)
        df_test[c]  = df_test[c].fillna(0)

    # LotFrontage: median by Neighborhood
    if "LotFrontage" in df_train.columns and "Neighborhood" in df_train.columns:
        train_med = df_train.groupby("Neighborhood")["LotFrontage"].median()
        df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        # test: use train medians; fallback to global median
        df_test["LotFrontage"] = df_test.apply(
            lambda row: train_med[row["Neighborhood"]] if pd.isnull(row["LotFrontage"]) and row["Neighborhood"] in train_med.index else row["LotFrontage"],
            axis=1
        )
        global_med = df_train["LotFrontage"].median()
        df_train["LotFrontage"].fillna(global_med, inplace=True)
        df_test["LotFrontage"].fillna(global_med, inplace=True)

    # Small categorical gaps -> fill with train mode
    small_mode_cols = [c for c in df_train.columns if df_train[c].isnull().sum() > 0 and df_train[c].dtype == "object"]
    # remove none_cols from this list
    small_mode_cols = [c for c in small_mode_cols if c not in none_cols]
    for c in small_mode_cols:
        mode_val = df_train[c].mode(dropna=True)[0]
        df_train[c].fillna(mode_val, inplace=True)
        df_test[c].fillna(mode_val, inplace=True)

    return df_train, df_test

def map_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    # Apply quality mapping where present
    for col in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond"]:
        if col in df.columns:
            df[col] = df[col].replace(QUALITY_MAP).fillna(0).astype(int)
    # Example: map overall condition if available (names may differ)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Example derived features
    if {"TotalBsmtSF","1stFlrSF","2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    if "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"] if "YrSold" in df.columns else 0
    # Binary flags
    for col in ["PoolQC","MiscFeature","Alley","Fence"]:
        if col in df.columns:
            df[f"Has_{col}"] = (~df[col].isnull()) & (df[col] != "None")
    return df

def clean_data(df_train: pd.DataFrame, df_test: pd.DataFrame, drop_threshold: float = 0.5) -> Dict:
    """
    Clean train/test in place and return a dict with cleaned frames and metadata.
    """
    # Copy to avoid accidental external changes
    train = df_train.copy()
    test  = df_test.copy()

    train, test, dropped = drop_sparse_columns(train, test, threshold=drop_threshold)
    train, test = impute_by_rules(train, test)
    train = map_ordinals(train)
    test  = map_ordinals(test)
    train = feature_engineering(train)
    test  = feature_engineering(test)

    # Final safety: fill any remaining numeric NAs with median, categoricals with "None"
    for c in train.select_dtypes(include=[np.number]).columns:
        if train[c].isnull().any():
            med = train[c].median()
            train[c].fillna(med, inplace=True)
            if c in test.columns:
                test[c].fillna(med, inplace=True)
    for c in train.select_dtypes(include=["object"]).columns:
        if train[c].isnull().any():
            train[c].fillna("None", inplace=True)
            if c in test.columns:
                test[c].fillna("None", inplace=True)

    # Verify
    missing_train = train.isnull().sum().sum()
    missing_test = test.isnull().sum().sum()

    return {
        "train": train,
        "test": test,
        "dropped_columns": dropped,
        "missing_after_train": int(missing_train),
        "missing_after_test": int(missing_test)
    }
