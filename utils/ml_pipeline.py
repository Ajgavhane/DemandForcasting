"""
ML Prediction Pipeline — Retail Demand Forecasting System
Handles feature engineering, model loading, and inference.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match training pipeline exactly
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "backend" / "models" / "demand_model.pkl"

FEATURE_COLS = [
    "category_encoded",          # 0 = Cold Drinks & Juices, 1 = Snacks & Munchies
    "price",                     # unit selling price
    "promotion_flag",            # active promotion (0/1)
    "campaign_active",           # marketing campaign (0/1)
    "total_spend",               # marketing spend / marketing_intensity proxy
    "net_stock",                 # current stock level
    "stock_turnover_rate",       # how fast stock sells
    "weekend_flag",              # weekend (0/1)
    "festival_flag",             # festival period (0/1)
    "lag_1",                     # previous day demand
    "lag_7",                     # demand 7 days ago
    "rolling_mean_7",            # 7-day average demand
    "rolling_std_7",             # 7-day demand std dev
    "cross_product_demand_signal",  # complementary category demand
]

IMPROVED_FEATURE_COLS = FEATURE_COLS + [
    "promo_intensity",           # promotion_flag × marketing_intensity
    "high_marketing_flag",       # binary: above-median marketing spend
    "demand_ratio",              # beverage / snack demand ratio
]

# Category encoding map
CATEGORY_ENCODING = {
    "Cold Drinks & Juices": 0,
    "Snacks & Munchies":    1,
}

# Product catalogue with defaults (used when product selected from dropdown)
PRODUCT_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "Cola":          {"category": "Cold Drinks & Juices", "price": 40.0},
    "Mango Juice":   {"category": "Cold Drinks & Juices", "price": 35.0},
    "Lemon Water":   {"category": "Cold Drinks & Juices", "price": 20.0},
    "Orange Soda":   {"category": "Cold Drinks & Juices", "price": 30.0},
    "Chips":         {"category": "Snacks & Munchies",    "price": 20.0},
    "Popcorn":       {"category": "Snacks & Munchies",    "price": 25.0},
    "Namkeen":       {"category": "Snacks & Munchies",    "price": 15.0},
    "Biscuits":      {"category": "Snacks & Munchies",    "price": 10.0},
    "Wafers":        {"category": "Snacks & Munchies",    "price": 20.0},
    "Energy Drink":  {"category": "Cold Drinks & Juices", "price": 80.0},
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

_model_cache: Optional[Any] = None  # in-memory cache


def load_model(path: Optional[Path] = None) -> Any:
    """
    Load the trained XGBoost model from disk with caching.

    Args:
        path: Override model file path.

    Returns:
        Trained XGBoost regressor.

    Raises:
        FileNotFoundError: If no model file found.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model_file = path or MODEL_PATH
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_file}. "
            "Run train_model.py first to generate demand_model.pkl."
        )

    with open(model_file, "rb") as f:
        _model_cache = pickle.load(f)

    return _model_cache


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw user inputs into the full feature vector for prediction.

    Args:
        input_data: Dict with keys matching the dashboard sidebar inputs.

    Returns:
        Single-row DataFrame with columns = IMPROVED_FEATURE_COLS.
    """
    d = input_data

    category = d.get("category", "Cold Drinks & Juices")
    category_encoded = CATEGORY_ENCODING.get(category, 0)

    promotion_flag   = int(d.get("promotion_flag", 0))
    campaign_active  = int(d.get("campaign_active", 0))
    marketing_spend  = float(d.get("total_spend", 0.5))  # normalised 0–1

    lag_1           = float(d.get("lag_1", 50.0))
    lag_7           = float(d.get("lag_7", 50.0))
    rolling_mean_7  = float(d.get("rolling_mean_7", 50.0))
    rolling_std_7   = float(d.get("rolling_std_7", 5.0))
    net_stock       = float(d.get("net_stock", 100.0))

    # derived promotion features (mirrors training pipeline)
    promo_intensity     = promotion_flag * marketing_spend
    marketing_median    = 0.5  # approximate median from training set
    high_marketing_flag = int(marketing_spend > marketing_median)

    # cross-category demand ratio (use provided or default balanced)
    demand_ratio = float(d.get("demand_ratio", 1.0))

    # stock turnover = sales / max(stock, 1)
    stock_turnover_rate = rolling_mean_7 / max(net_stock, 1.0)

    # cross-product signal: use the opposite-category daily demand estimate
    cross_product_demand_signal = float(d.get("cross_product_demand_signal", rolling_mean_7))

    row = {
        "category_encoded":            category_encoded,
        "price":                       float(d.get("price", 30.0)),
        "promotion_flag":              promotion_flag,
        "campaign_active":             campaign_active,
        "total_spend":                 marketing_spend,
        "net_stock":                   net_stock,
        "stock_turnover_rate":         round(stock_turnover_rate, 4),
        "weekend_flag":                int(d.get("weekend_flag", 0)),
        "festival_flag":               int(d.get("festival_flag", 0)),
        "lag_1":                       lag_1,
        "lag_7":                       lag_7,
        "rolling_mean_7":              rolling_mean_7,
        "rolling_std_7":               rolling_std_7,
        "cross_product_demand_signal": cross_product_demand_signal,
        "promo_intensity":             promo_intensity,
        "high_marketing_flag":         high_marketing_flag,
        "demand_ratio":                demand_ratio,
    }

    return pd.DataFrame([row])


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT DEMAND
# ──────────────────────────────────────────────────────────────────────────────

def predict_demand(
    input_data: Dict[str, Any],
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run end-to-end demand prediction for a single input row.

    Args:
        input_data: Raw user / API input dict.
        model: Pre-loaded model (optional, will auto-load if None).

    Returns:
        Dict with predicted_demand + feature values used.
    """
    m = model or load_model()
    feature_df = engineer_features(input_data)

    # Use improved features if available in model
    try:
        cols = IMPROVED_FEATURE_COLS
        raw = np.clip(m.predict(feature_df[cols]), 0, None)
    except Exception:
        # Fall back to base features
        cols = FEATURE_COLS
        raw = np.clip(m.predict(feature_df[cols]), 0, None)

    predicted = round(float(raw[0]), 2)

    return {
        "predicted_demand": predicted,
        "features_used":    feature_df.to_dict(orient="records")[0],
    }


# ──────────────────────────────────────────────────────────────────────────────
# BATCH FORECAST (DataFrame input)
# ──────────────────────────────────────────────────────────────────────────────

def batch_predict(df: pd.DataFrame, model: Optional[Any] = None) -> np.ndarray:
    """
    Run batch predictions on a preprocessed DataFrame.

    Args:
        df: DataFrame with columns matching IMPROVED_FEATURE_COLS.
        model: Pre-loaded model.

    Returns:
        NumPy array of predicted demand values (≥ 0).
    """
    m = model or load_model()
    try:
        preds = np.clip(m.predict(df[IMPROVED_FEATURE_COLS]), 0, None)
    except Exception:
        preds = np.clip(m.predict(df[FEATURE_COLS]), 0, None)
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model: Optional[Any] = None) -> Dict[str, float]:
    """
    Extract feature importance from the trained model.

    Returns:
        Dict mapping feature_name → importance_score, sorted descending.
    """
    m = model or load_model()
    importances = m.feature_importances_

    # Determine which feature list the model was trained on
    n = len(importances)
    if n == len(IMPROVED_FEATURE_COLS):
        cols = IMPROVED_FEATURE_COLS
    else:
        cols = FEATURE_COLS[:n]

    fi = dict(zip(cols, importances.tolist()))
    return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
