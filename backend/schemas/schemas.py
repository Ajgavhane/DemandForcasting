"""
Pydantic Schemas — Retail Demand Forecasting API
Request and response validation models.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-row demand prediction request."""

    product_name:               str   = Field(..., example="Cola")
    category:                   str   = Field("Cold Drinks & Juices",
                                               example="Cold Drinks & Juices")
    price:                      float = Field(30.0, ge=1.0,  le=1000.0, example=40.0)
    promotion_flag:             int   = Field(0,    ge=0,    le=1)
    campaign_active:            int   = Field(0,    ge=0,    le=1)
    total_spend:                float = Field(0.5,  ge=0.0,  le=1.0,
                                               description="Normalised marketing intensity 0–1")
    net_stock:                  float = Field(100.0, ge=0.0, example=80.0)
    weekend_flag:               int   = Field(0,    ge=0,    le=1)
    festival_flag:              int   = Field(0,    ge=0,    le=1)
    lag_1:                      float = Field(50.0, ge=0.0,  example=55.0)
    lag_7:                      float = Field(50.0, ge=0.0,  example=48.0)
    rolling_mean_7:             float = Field(50.0, ge=0.0,  example=52.0)
    rolling_std_7:              float = Field(5.0,  ge=0.0,  example=4.0)
    cross_product_demand_signal: float = Field(50.0, ge=0.0)
    demand_ratio:               float = Field(1.0,  ge=0.0)

    class Config:
        json_schema_extra = {
            "example": {
                "product_name":   "Cola",
                "category":       "Cold Drinks & Juices",
                "price":          40.0,
                "promotion_flag": 1,
                "campaign_active": 1,
                "total_spend":    0.8,
                "net_stock":      60.0,
                "weekend_flag":   1,
                "festival_flag":  0,
                "lag_1":          62.0,
                "lag_7":          55.0,
                "rolling_mean_7": 58.0,
                "rolling_std_7":  6.0,
                "cross_product_demand_signal": 48.0,
                "demand_ratio":   1.2,
            }
        }


class ForecastRequest(BaseModel):
    """Multi-row batch forecast request (CSV upload or API)."""

    rows: List[PredictRequest] = Field(..., min_length=1, max_length=500)


class InventoryRiskRequest(BaseModel):
    """Inventory risk assessment without requiring a model prediction."""

    product_name:   str   = Field(..., example="Cola")
    current_stock:  float = Field(..., ge=0.0, example=60.0)
    avg_demand:     float = Field(..., ge=0.1, example=55.0)
    demand_std:     float = Field(..., ge=0.0, example=6.0)
    predicted_demand: float = Field(..., ge=0.0, example=62.0)
    rolling_mean_7: float = Field(..., ge=0.0, example=58.0)
    weekend_flag:   int   = Field(0, ge=0, le=1)
    festival_flag:  int   = Field(0, ge=0, le=1)
    promotion_flag: int   = Field(0, ge=0, le=1)


# ──────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

class InventoryMetrics(BaseModel):
    safety_stock:       float
    reorder_point:      float
    reorder_quantity:   float
    stockout_risk_pct:  float
    health_score:       float
    alert_code:         str
    alert_label:        str
    recommendations:    List[str]


class PredictResponse(BaseModel):
    product_name:       str
    predicted_demand:   float
    inventory:          InventoryMetrics
    features_used:      dict


class ForecastRow(BaseModel):
    product_name:       str
    predicted_demand:   float
    stockout_risk_pct:  float
    alert_label:        str
    reorder_quantity:   float


class ForecastResponse(BaseModel):
    total_rows:         int
    rows:               List[ForecastRow]


class InventoryRiskResponse(BaseModel):
    product_name:       str
    safety_stock:       float
    reorder_point:      float
    reorder_quantity:   float
    stockout_risk_pct:  float
    health_score:       float
    alert_code:         str
    alert_label:        str
    recommendations:    List[str]


class FeatureImportanceItem(BaseModel):
    feature:            str
    importance:         float


class FeatureImportanceResponse(BaseModel):
    features:           List[FeatureImportanceItem]


class KPISummary(BaseModel):
    total_products:         int
    stockout_count:         int
    overstock_count:        int
    safe_count:             int
    avg_health_score:       float
    avg_stockout_risk_pct:  float
    avg_predicted_demand:   float
    baseline_mae:           float
    improved_mae:           float


class DashboardSummaryResponse(BaseModel):
    kpis:                   KPISummary
    recent_alerts:          List[dict]
    top_risk_products:      List[dict]
    model_info:             dict
