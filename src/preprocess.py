# src/preprocess.py
"""
Professional, robust cleaning for Ames dataset.
- drops very sparse columns
- imputes by clear rules
- uses train statistics to fill test
- avoids chained-assignment and pandas warnings
"""

from typing import Dict
import pandas as pd
import numpy as np


QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}


def drop_sparse_columns(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.5):
    frac = train.isnull().mean()
    to_drop = frac[frac > threshold].index.tolist()
    train = train.drop(columns=to_drop, errors="ignore").copy()
    test = test.drop(columns=[c for c in to_drop if c in test.columns], errors="ignore").copy()
    return train, test, to_drop


def impute_by_rules(train: pd.DataFrame, test: pd.DataFrame):
    # work on copies
    tr = train.copy()
    te = test.copy()

    # 1) Keep MasVnrType + MasVnrArea
    if "MasVnrType" in tr.columns:
        tr.loc[:, "MasVnrType"] = tr["MasVnrType"].fillna("None")
        te.loc[:, "MasVnrType"] = te["MasVnrType"].fillna("None")
    if "MasVnrArea" in tr.columns:
        tr.loc[:, "MasVnrArea"] = tr["MasVnrArea"].fillna(0)
        te.loc[:, "MasVnrArea"] = te["MasVnrArea"].fillna(0)

    # 2) Categorical 'absence' features -> "None"
    none_keywords = ["Garage", "Bsmt", "Fireplace", "Pool", "Alley", "Fence", "Misc"]
    cat_with_na = [c for c in tr.select_dtypes(include=["object"]).columns if tr[c].isnull().any()]
    none_cols = [c for c in cat_with_na if any(k in c for k in none_keywords)]
    for c in none_cols:
        tr.loc[:, c] = tr[c].fillna("None")
        if c in te.columns:
            te.loc[:, c] = te[c].fillna("None")

    # 3) Numeric 'absence' features -> 0 (by keyword)
    num_with_na = [c for c in tr.select_dtypes(include=[np.number]).columns if tr[c].isnull().any()]
    zero_keywords = ["Area", "SF", "Cars", "Bath", "YrBlt"]
    zero_cols = [c for c in num_with_na if any(k in c for k in zero_keywords)]
    for c in zero_cols:
        tr.loc[:, c] = tr[c].fillna(0)
        if c in te.columns:
            te.loc[:, c] = te[c].fillna(0)

    # 4) LotFrontage: neighborhood median from train, fallback to global median
    if "LotFrontage" in tr.columns and "Neighborhood" in tr.columns:
        neigh_meds = tr.groupby("Neighborhood")["LotFrontage"].median()
        tr.loc[:, "LotFrontage"] = tr.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        # test: map where neighborhood known, then fallback to train global median
        if "LotFrontage" in te.columns and "Neighborhood" in te.columns:
            # rows where LotFrontage is missing and neighborhood exists in train medians
            mask = te["LotFrontage"].isna()
            mask_map = mask & te["Neighborhood"].isin(neigh_meds.index)
            te.loc[mask_map, "LotFrontage"] = te.loc[mask_map, "Neighborhood"].map(neigh_meds)
            global_med = tr["LotFrontage"].median()
            te.loc[te["LotFrontage"].isna(), "LotFrontage"] = global_med
        # ensure train has no NaN left
        tr.loc[:, "LotFrontage"] = tr["LotFrontage"].fillna(tr["LotFrontage"].median())

    # 5) Small categorical gaps -> fill with train mode
    still_cat_na = [c for c in tr.columns if tr[c].dtype == "object" and tr[c].isnull().any()]
    for c in still_cat_na:
        mode_val = tr[c].mode(dropna=True)[0] if not tr[c].mode(dropna=True).empty else "None"
        tr.loc[:, c] = tr[c].fillna(mode_val)
        if c in te.columns:
            te.loc[:, c] = te[c].fillna(mode_val)

    # 6) Final numeric safety: fill any remaining numeric NaNs in test with train median
    num_cols = tr.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if c in te.columns and te[c].isnull().any():
            med = tr[c].median()
            te.loc[te[c].isnull(), c] = med
        # ensure train filled
        if tr[c].isnull().any():
            tr.loc[:, c] = tr[c].fillna(tr[c].median())

    return tr, te


def map_ordinals(df: pd.DataFrame):
    df = df.copy()
    ordinal_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    for col in ordinal_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].replace(QUALITY_MAP).fillna(0).astype(int)
    return df


def feature_engineering(df: pd.DataFrame):
    df = df.copy()
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df.loc[:, "TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    if "YearBuilt" in df.columns and "YrSold" in df.columns:
        df.loc[:, "HouseAge"] = df["YrSold"] - df["YearBuilt"]
    return df


def clean_data(df_train: pd.DataFrame, df_test: pd.DataFrame, drop_threshold: float = 0.5) -> Dict:
    """
    Run full cleaning pipeline and return a dict with cleaned frames + metadata.
    """
    # copy inputs to avoid side-effects
    train = df_train.copy()
    test = df_test.copy()

    train, test, dropped = drop_sparse_columns(train, test, threshold=drop_threshold)
    train, test = impute_by_rules(train, test)
    train = map_ordinals(train)
    test = map_ordinals(test)
    train = feature_engineering(train)
    test = feature_engineering(test)

    missing_train = int(train.isnull().sum().sum())
    missing_test = int(test.isnull().sum().sum())

    return {
        "train": train,
        "test": test,
        "dropped_columns": dropped,
        "missing_after_train": missing_train,
        "missing_after_test": missing_test
    }
