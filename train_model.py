"""
Model Training Script — Retail Demand Forecasting System
Trains XGBoost on the augmented dataset and saves demand_model.pkl.

Usage:
    python train_model.py --data path/to/augmented_ml_ready_dataset.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODEL_OUTPUT = ROOT / "backend" / "models" / "demand_model.pkl"

SPLIT_RATIO  = 0.80
LEAD_TIME    = 3
Z_SCORE      = 1.65
SPIKE_MULT   = 1.5

BEVERAGE_COL = "category_Cold Drinks & Juices"
SNACK_COL    = "category_Snacks & Munchies"

IMPROVED_FEATURE_COLS = [
    "category_encoded", "price", "promotion_flag", "campaign_active",
    "total_spend", "net_stock", "stock_turnover_rate", "weekend_flag",
    "festival_flag", "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7",
    "cross_product_demand_signal",
    "promo_intensity", "high_marketing_flag", "demand_ratio",
]

TARGET = "quantity_sold"


def load_and_preprocess(file_path: str) -> pd.DataFrame:
    print(f"[1/6] Loading data from {file_path}")
    raw = pd.read_csv(file_path)

    df = raw.copy()
    df["order_date"] = pd.to_datetime(df["date"])
    df = df.sort_values("order_date").reset_index(drop=True)

    # Category filter
    target_mask = (df[BEVERAGE_COL] == 1) | (df[SNACK_COL] == 1)
    df = df[target_mask].copy().reset_index(drop=True)

    df["category"] = np.where(df[BEVERAGE_COL] == 1,
                               "Cold Drinks & Juices", "Snacks & Munchies")
    df["category_encoded"] = np.where(df[BEVERAGE_COL] == 1, 0, 1)

    # Missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Drop leakage
    for c in ["reorder_quantity", "low_stock_alert", "stock_status"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Rename rolling columns
    df = df.rename(columns={"rolling_7_mean": "rolling_mean_7",
                             "rolling_7_std":  "rolling_std_7"})

    # Drop NaN lag rows
    df = df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"])

    if "price" not in df.columns and "avg_unit_price" in df.columns:
        df["price"] = df["avg_unit_price"]
    if "total_spend" not in df.columns:
        df["total_spend"] = df.get("marketing_intensity", 0.5)

    print(f"   Shape after preprocessing: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Engineering features")

    # Cross-product demand signal
    daily_cat = (
        df.groupby(["order_date", "category"])["quantity_sold"]
        .sum().reset_index()
    )
    cat_pivot = daily_cat.pivot(
        index="order_date", columns="category", values="quantity_sold"
    ).reset_index().fillna(0)
    cat_pivot.columns.name = None
    bev_col = "Cold Drinks & Juices"
    snk_col = "Snacks & Munchies"

    df = df.merge(
        cat_pivot[["order_date", bev_col, snk_col]],
        on="order_date", how="left", suffixes=("", "_daily")
    )
    df["cross_product_demand_signal"] = np.where(
        df["category"] == "Snacks & Munchies", df[bev_col], df[snk_col]
    )
    df = df.drop(columns=[bev_col, snk_col], errors="ignore")

    # Promotion features
    df["promo_intensity"]    = df["promotion_flag"] * df["marketing_intensity"]
    mkt_median               = df["marketing_intensity"].median()
    df["high_marketing_flag"] = (df["marketing_intensity"] > mkt_median).astype(int)

    # Daily demand per category
    daily_bev = (
        daily_cat[daily_cat["category"] == "Cold Drinks & Juices"]
        [["order_date", "quantity_sold"]]
        .rename(columns={"quantity_sold": "beverage_demand_per_day"})
    )
    daily_snk = (
        daily_cat[daily_cat["category"] == "Snacks & Munchies"]
        [["order_date", "quantity_sold"]]
        .rename(columns={"quantity_sold": "snack_demand_per_day"})
    )
    df = df.merge(daily_bev, on="order_date", how="left")
    df = df.merge(daily_snk, on="order_date", how="left")
    df["demand_ratio"] = (
        df["beverage_demand_per_day"] / (df["snack_demand_per_day"] + 1)
    )

    df["stock_turnover_rate"] = df["rolling_mean_7"] / df["net_stock"].clip(lower=1.0)

    print(f"   Features engineered. Shape: {df.shape}")
    return df


def train(df: pd.DataFrame) -> tuple:
    print("[3/6] Training XGBoost model")

    X = df[IMPROVED_FEATURE_COLS]
    y = df[TARGET]

    split_idx = int(len(df) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 0, None)
    mae    = mean_absolute_error(y_test, y_pred)

    print(f"   Test MAE: {mae:.4f} units")
    return model, mae


def save_model(model: object, path: Path) -> None:
    print(f"[4/6] Saving model to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"   Saved ✓")


def main(data_path: str) -> None:
    df    = load_and_preprocess(data_path)
    df    = engineer_features(df)
    model, mae = train(df)
    save_model(model, MODEL_OUTPUT)
    print(f"\n{'='*50}")
    print(f"  Training complete — MAE: {mae:.4f} units")
    print(f"  Model saved → {MODEL_OUTPUT}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand forecasting model")
    parser.add_argument(
        "--data",
        default="data/augmented_ml_ready_dataset.csv",
        help="Path to the training CSV",
    )
    args = parser.parse_args()
    main(args.data)
