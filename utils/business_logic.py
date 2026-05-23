"""
Business Logic Engine — Retail Demand Forecasting System
Handles inventory calculations, risk scoring, and recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
LEAD_TIME_DAYS: int = 3       # supplier replenishment lead time
Z_SCORE_95: float = 1.65      # 95% service level
SPIKE_MULTIPLIER: float = 1.5 # demand spike threshold (150% of rolling avg)
EMA_ALPHA: float = 0.3        # exponential moving average smoothing factor


# ──────────────────────────────────────────────────────────────────────────────
# SAFETY STOCK
# ──────────────────────────────────────────────────────────────────────────────

def calculate_safety_stock(
    demand_std: float,
    lead_time: int = LEAD_TIME_DAYS,
    service_level_z: float = Z_SCORE_95,
) -> float:
    """
    Safety Stock = Z × σ_demand × √(lead_time)

    Args:
        demand_std: Standard deviation of daily demand (rolling_std_7).
        lead_time: Supplier lead time in days.
        service_level_z: Z-score for desired service level (1.65 = 95%).

    Returns:
        Safety stock quantity (units, ≥ 0).
    """
    ss = service_level_z * max(float(demand_std), 0.0) * np.sqrt(lead_time)
    return round(max(ss, 0.0), 2)


# ──────────────────────────────────────────────────────────────────────────────
# REORDER POINT
# ──────────────────────────────────────────────────────────────────────────────

def calculate_reorder_point(
    avg_demand: float,
    demand_std: float,
    lead_time: int = LEAD_TIME_DAYS,
    service_level_z: float = Z_SCORE_95,
) -> float:
    """
    Reorder Point = μ_demand × lead_time + safety_stock

    Args:
        avg_demand: Average daily demand (rolling_mean_7).
        demand_std: Std dev of daily demand.
        lead_time: Supplier lead time in days.
        service_level_z: Z-score for service level.

    Returns:
        Reorder point in units.
    """
    ss = calculate_safety_stock(demand_std, lead_time, service_level_z)
    rp = max(float(avg_demand), 0.1) * lead_time + ss
    return round(rp, 2)


# ──────────────────────────────────────────────────────────────────────────────
# REORDER QUANTITY
# ──────────────────────────────────────────────────────────────────────────────

def calculate_reorder_quantity(
    predicted_demand: float,
    current_stock: float,
    safety_stock: float,
    lead_time: int = LEAD_TIME_DAYS,
) -> float:
    """
    Reorder Quantity = (predicted_demand × lead_time + safety_stock) - current_stock
    Clipped to zero if stock is already adequate.

    Returns:
        Units to order (≥ 0).
    """
    target = predicted_demand * lead_time + safety_stock
    qty = target - max(float(current_stock), 0.0)
    return round(max(qty, 0.0), 2)


# ──────────────────────────────────────────────────────────────────────────────
# STOCKOUT RISK SCORE (0–100)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_stockout_risk_score(
    current_stock: float,
    reorder_point: float,
    predicted_demand: float,
) -> float:
    """
    Stockout Risk Score maps the gap between stock and reorder point
    to a 0–100 % scale.

    Score = 100 × (1 − current_stock / reorder_point)   clipped [0, 100]

    Returns:
        Risk percentage (0 = no risk, 100 = definite stockout).
    """
    if reorder_point <= 0:
        return 0.0
    ratio = float(current_stock) / float(reorder_point)
    score = (1.0 - ratio) * 100.0
    return round(float(np.clip(score, 0.0, 100.0)), 1)


# ──────────────────────────────────────────────────────────────────────────────
# INVENTORY HEALTH SCORE (0–100)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_inventory_health_score(
    current_stock: float,
    reorder_point: float,
    predicted_demand: float,
    demand_std: float,
) -> float:
    """
    Composite inventory health score.

    Components:
        - Coverage ratio  (40 pts): days of cover relative to 7-day target
        - Stockout buffer (40 pts): buffer above reorder point
        - Volatility      (20 pts): lower std = higher score

    Returns:
        Health score 0–100.
    """
    # coverage component
    daily_demand = max(predicted_demand, 0.1)
    days_cover = float(current_stock) / daily_demand
    target_days = 7.0
    coverage_score = min(days_cover / target_days, 1.0) * 40.0

    # buffer above reorder point
    buffer = float(current_stock) - float(reorder_point)
    buffer_score = min(max(buffer, 0.0) / max(reorder_point, 1.0), 1.0) * 40.0

    # volatility component (lower cv = better)
    cv = float(demand_std) / max(daily_demand, 0.1)
    volatility_score = max(1.0 - cv / 2.0, 0.0) * 20.0

    total = coverage_score + buffer_score + volatility_score
    return round(float(np.clip(total, 0.0, 100.0)), 1)


# ──────────────────────────────────────────────────────────────────────────────
# ALERT CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def classify_alert(
    current_stock: float,
    reorder_point: float,
    predicted_demand: float,
    rolling_mean_7: float,
) -> Tuple[str, str]:
    """
    Returns (alert_code, alert_label) tuple.

    Codes: STOCKOUT | OVERSTOCK | SPIKE | OK
    """
    stockout  = current_stock < reorder_point
    overstock = current_stock > 2.0 * reorder_point
    spike     = predicted_demand > rolling_mean_7 * SPIKE_MULTIPLIER

    if stockout and spike:
        return "STOCKOUT", "🔴 CRITICAL: Stockout + Demand Spike"
    if stockout:
        return "STOCKOUT", "🔴 STOCKOUT RISK"
    if overstock:
        return "OVERSTOCK", "🟡 OVERSTOCK"
    if spike:
        return "SPIKE", "⚡ DEMAND SPIKE"
    return "OK", "🟢 SAFE"


# ──────────────────────────────────────────────────────────────────────────────
# TREND & VOLATILITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def compute_demand_trend(demand_series: pd.Series) -> Dict[str, Any]:
    """
    Compute trend, EMA, and volatility stats from a demand series.

    Returns:
        dict with keys: ema, rolling_mean_7, rolling_std_7,
                        trend_direction, trend_pct_change
    """
    s = demand_series.dropna().astype(float)
    if len(s) < 2:
        return {}

    ema = s.ewm(alpha=EMA_ALPHA, adjust=False).mean()
    rm7 = s.rolling(7, min_periods=1).mean()
    rs7 = s.rolling(7, min_periods=1).std().fillna(0)

    trend_pct = (
        (s.iloc[-1] - s.iloc[0]) / max(abs(s.iloc[0]), 0.01)
    ) * 100.0

    direction = "up" if trend_pct > 2 else ("down" if trend_pct < -2 else "flat")

    return {
        "ema":              ema.tolist(),
        "rolling_mean_7":   rm7.tolist(),
        "rolling_std_7":    rs7.tolist(),
        "trend_direction":  direction,
        "trend_pct_change": round(trend_pct, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# AI RECOMMENDATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def generate_recommendation(
    product_name: str,
    alert_code: str,
    predicted_demand: float,
    current_stock: float,
    reorder_quantity: float,
    stockout_risk_pct: float,
    health_score: float,
    weekend_flag: int = 0,
    festival_flag: int = 0,
    promotion_flag: int = 0,
) -> List[str]:
    """
    Generate business-level actionable recommendations.

    Returns:
        List of recommendation strings (1-4 items).
    """
    recs: List[str] = []

    if alert_code == "STOCKOUT":
        recs.append(
            f"🚨 Order {reorder_quantity:.0f} units of {product_name} immediately "
            f"(stockout risk: {stockout_risk_pct:.0f}%)."
        )

    if weekend_flag and predicted_demand > 0:
        weekend_boost = round(predicted_demand * 0.15, 1)
        recs.append(
            f"📅 Weekend demand spike expected for {product_name}. "
            f"Buffer +{weekend_boost:.0f} units above base forecast."
        )

    if festival_flag and predicted_demand > 0:
        festival_boost = round(predicted_demand * 0.25, 1)
        recs.append(
            f"🎉 Festival period active — consider stocking {festival_boost:.0f} "
            f"extra units to avoid {product_name} stockouts."
        )

    if promotion_flag and current_stock < predicted_demand * 2:
        recs.append(
            f"📣 Promotion is active for {product_name}. "
            f"Increase stock to ≥{predicted_demand * 2:.0f} units to support lift."
        )

    if alert_code == "OVERSTOCK":
        recs.append(
            f"📦 {product_name} is overstocked. Consider a clearance promotion "
            "or redistribute to high-demand locations."
        )

    if health_score < 40:
        recs.append(
            f"⚠️  {product_name} inventory health is low ({health_score:.0f}/100). "
            "Review reorder schedule with your supply chain team."
        )

    if not recs:
        recs.append(
            f"✅ {product_name} inventory is healthy (score: {health_score:.0f}/100). "
            "No immediate action required."
        )

    return recs


# ──────────────────────────────────────────────────────────────────────────────
# FULL ANALYTICS BUNDLE (single-row)
# ──────────────────────────────────────────────────────────────────────────────

def compute_inventory_analytics(
    product_name: str,
    predicted_demand: float,
    current_stock: float,
    rolling_mean_7: float,
    rolling_std_7: float,
    weekend_flag: int = 0,
    festival_flag: int = 0,
    promotion_flag: int = 0,
    lead_time: int = LEAD_TIME_DAYS,
) -> Dict[str, Any]:
    """
    Full analytics bundle for a single product row.

    Returns:
        Dictionary of all computed inventory metrics + alerts + recommendations.
    """
    safety_stock    = calculate_safety_stock(rolling_std_7, lead_time)
    reorder_point   = calculate_reorder_point(rolling_mean_7, rolling_std_7, lead_time)
    reorder_qty     = calculate_reorder_quantity(predicted_demand, current_stock, safety_stock, lead_time)
    stockout_risk   = calculate_stockout_risk_score(current_stock, reorder_point, predicted_demand)
    health_score    = calculate_inventory_health_score(current_stock, reorder_point, predicted_demand, rolling_std_7)
    alert_code, alert_label = classify_alert(current_stock, reorder_point, predicted_demand, rolling_mean_7)
    recommendations = generate_recommendation(
        product_name, alert_code, predicted_demand, current_stock,
        reorder_qty, stockout_risk, health_score,
        weekend_flag, festival_flag, promotion_flag,
    )

    return {
        "safety_stock":       safety_stock,
        "reorder_point":      reorder_point,
        "reorder_quantity":   reorder_qty,
        "stockout_risk_pct":  stockout_risk,
        "health_score":       health_score,
        "alert_code":         alert_code,
        "alert_label":        alert_label,
        "recommendations":    recommendations,
    }
