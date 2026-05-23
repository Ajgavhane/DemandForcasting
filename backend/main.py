"""
FastAPI Backend — Retail Demand Forecasting & Inventory Optimization API
All endpoints for prediction, forecasting, inventory risk, and dashboard summary.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add project root to sys.path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.db.models import (
    create_all_tables,
    get_db,
    PredictionRecord,
    AlertRecord,
    InventoryRecord,
)
from backend.schemas.schemas import (
    PredictRequest,
    PredictResponse,
    ForecastRequest,
    ForecastResponse,
    ForecastRow,
    InventoryRiskRequest,
    InventoryRiskResponse,
    FeatureImportanceResponse,
    FeatureImportanceItem,
    DashboardSummaryResponse,
    KPISummary,
    InventoryMetrics,
)
from utils.ml_pipeline import predict_demand, get_feature_importance, PRODUCT_CATALOGUE
from utils.business_logic import compute_inventory_analytics

# ──────────────────────────────────────────────────────────────────────────────
# APP INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "AI-powered retail demand forecasting and inventory optimization system. "
        "XGBoost model trained on Beverages & Snacks data — MAE 4.36 units."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Create DB tables on startup."""
    create_all_tables()


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "Retail Demand Forecasting API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(
    req: PredictRequest,
    db: Session = Depends(get_db),
):
    """
    Run demand prediction for a single product configuration.
    Returns predicted demand + full inventory analytics.
    """
    try:
        # Run prediction
        result = predict_demand(req.model_dump())
        pred   = result["predicted_demand"]

        # Inventory analytics
        analytics = compute_inventory_analytics(
            product_name   = req.product_name,
            predicted_demand = pred,
            current_stock  = req.net_stock,
            rolling_mean_7 = req.rolling_mean_7,
            rolling_std_7  = req.rolling_std_7,
            weekend_flag   = req.weekend_flag,
            festival_flag  = req.festival_flag,
            promotion_flag = req.promotion_flag,
        )

        # Persist to DB
        record = PredictionRecord(
            product_name      = req.product_name,
            category          = req.category,
            price             = req.price,
            promotion_flag    = bool(req.promotion_flag),
            festival_flag     = bool(req.festival_flag),
            weekend_flag      = bool(req.weekend_flag),
            current_stock     = req.net_stock,
            lag_1             = req.lag_1,
            lag_7             = req.lag_7,
            rolling_mean_7    = req.rolling_mean_7,
            rolling_std_7     = req.rolling_std_7,
            predicted_demand  = pred,
            safety_stock      = analytics["safety_stock"],
            reorder_point     = analytics["reorder_point"],
            reorder_quantity  = analytics["reorder_quantity"],
            stockout_risk_pct = analytics["stockout_risk_pct"],
            health_score      = analytics["health_score"],
            alert_code        = analytics["alert_code"],
            alert_label       = analytics["alert_label"],
            recommendations   = json.dumps(analytics["recommendations"]),
        )
        db.add(record)

        # Log alert if not OK
        if analytics["alert_code"] != "OK":
            alert = AlertRecord(
                product_name     = req.product_name,
                alert_code       = analytics["alert_code"],
                alert_label      = analytics["alert_label"],
                predicted_demand = pred,
                current_stock    = req.net_stock,
                reorder_point    = analytics["reorder_point"],
            )
            db.add(alert)

        db.commit()

        return PredictResponse(
            product_name     = req.product_name,
            predicted_demand = pred,
            inventory        = InventoryMetrics(**analytics),
            features_used    = result["features_used"],
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# POST /forecast — batch
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/forecast", response_model=ForecastResponse, tags=["Prediction"])
def forecast_endpoint(req: ForecastRequest):
    """
    Batch demand forecasting for multiple product-date rows.
    """
    rows: List[ForecastRow] = []
    for item in req.rows:
        try:
            result    = predict_demand(item.model_dump())
            pred      = result["predicted_demand"]
            analytics = compute_inventory_analytics(
                product_name     = item.product_name,
                predicted_demand = pred,
                current_stock    = item.net_stock,
                rolling_mean_7   = item.rolling_mean_7,
                rolling_std_7    = item.rolling_std_7,
                weekend_flag     = item.weekend_flag,
                festival_flag    = item.festival_flag,
                promotion_flag   = item.promotion_flag,
            )
            rows.append(ForecastRow(
                product_name      = item.product_name,
                predicted_demand  = pred,
                stockout_risk_pct = analytics["stockout_risk_pct"],
                alert_label       = analytics["alert_label"],
                reorder_quantity  = analytics["reorder_quantity"],
            ))
        except Exception:
            pass  # skip failed rows silently in batch mode

    return ForecastResponse(total_rows=len(rows), rows=rows)


# ──────────────────────────────────────────────────────────────────────────────
# POST /inventory-risk
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/inventory-risk", response_model=InventoryRiskResponse, tags=["Inventory"])
def inventory_risk_endpoint(req: InventoryRiskRequest):
    """
    Calculate inventory risk metrics without running ML prediction.
    Useful for scenarios where demand is already known.
    """
    analytics = compute_inventory_analytics(
        product_name     = req.product_name,
        predicted_demand = req.predicted_demand,
        current_stock    = req.current_stock,
        rolling_mean_7   = req.avg_demand,
        rolling_std_7    = req.demand_std,
        weekend_flag     = req.weekend_flag,
        festival_flag    = req.festival_flag,
        promotion_flag   = req.promotion_flag,
    )
    return InventoryRiskResponse(product_name=req.product_name, **analytics)


# ──────────────────────────────────────────────────────────────────────────────
# GET /feature-importance
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Model"])
def feature_importance_endpoint():
    """
    Return feature importances from the trained XGBoost model.
    """
    try:
        fi = get_feature_importance()
        items = [FeatureImportanceItem(feature=k, importance=v) for k, v in fi.items()]
        return FeatureImportanceResponse(features=items)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# GET /dashboard-summary
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/dashboard-summary", response_model=DashboardSummaryResponse, tags=["Dashboard"])
def dashboard_summary_endpoint(db: Session = Depends(get_db)):
    """
    Aggregated KPIs for the main dashboard.
    Reads from the predictions and alerts tables.
    """
    records = db.query(PredictionRecord).all()

    if not records:
        # Return empty placeholder when no predictions yet
        return DashboardSummaryResponse(
            kpis=KPISummary(
                total_products=0, stockout_count=0, overstock_count=0,
                safe_count=0, avg_health_score=0.0, avg_stockout_risk_pct=0.0,
                avg_predicted_demand=0.0, baseline_mae=4.8769, improved_mae=4.3601,
            ),
            recent_alerts=[],
            top_risk_products=[],
            model_info={"model": "XGBoost", "baseline_mae": 4.8769, "improved_mae": 4.3601},
        )

    total     = len(records)
    stockout  = sum(1 for r in records if r.alert_code == "STOCKOUT")
    overstock = sum(1 for r in records if r.alert_code == "OVERSTOCK")
    safe      = sum(1 for r in records if r.alert_code == "OK")
    avg_health = float(np.mean([r.health_score or 0 for r in records]))
    avg_risk   = float(np.mean([r.stockout_risk_pct or 0 for r in records]))
    avg_demand = float(np.mean([r.predicted_demand for r in records]))

    recent_alerts = [
        {
            "product": r.product_name,
            "alert":   r.alert_label,
            "demand":  r.predicted_demand,
            "stock":   r.current_stock,
        }
        for r in sorted(records, key=lambda x: x.created_at or 0, reverse=True)[:10]
        if r.alert_code != "OK"
    ]

    top_risk = sorted(records, key=lambda x: x.stockout_risk_pct or 0, reverse=True)[:5]
    top_risk_products = [
        {
            "product": r.product_name,
            "risk":    r.stockout_risk_pct,
            "stock":   r.current_stock,
            "demand":  r.predicted_demand,
        }
        for r in top_risk
    ]

    return DashboardSummaryResponse(
        kpis=KPISummary(
            total_products        = total,
            stockout_count        = stockout,
            overstock_count       = overstock,
            safe_count            = safe,
            avg_health_score      = round(avg_health, 1),
            avg_stockout_risk_pct = round(avg_risk, 1),
            avg_predicted_demand  = round(avg_demand, 2),
            baseline_mae          = 4.8769,
            improved_mae          = 4.3601,
        ),
        recent_alerts      = recent_alerts,
        top_risk_products  = top_risk_products,
        model_info={
            "model":        "XGBoost Regressor",
            "baseline_mae": 4.8769,
            "improved_mae": 4.3601,
            "improvement":  "10.6%",
            "features":     17,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /products
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/products", tags=["Catalogue"])
def list_products():
    """Return available product catalogue."""
    return {"products": PRODUCT_CATALOGUE}


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
