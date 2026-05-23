"""
SQLAlchemy Database Models — Retail Demand Forecasting System
Stores predictions, inventory records, alerts, and forecast history.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ──────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ──────────────────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./retail_forecast.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────

class PredictionRecord(Base):
    """Stores individual demand predictions from the API."""

    __tablename__ = "predictions"

    id                  = Column(Integer, primary_key=True, index=True)
    created_at          = Column(DateTime, default=datetime.utcnow, index=True)
    product_name        = Column(String(120), index=True, nullable=False)
    category            = Column(String(80))
    price               = Column(Float)
    promotion_flag      = Column(Boolean, default=False)
    festival_flag       = Column(Boolean, default=False)
    weekend_flag        = Column(Boolean, default=False)
    current_stock       = Column(Float)
    lag_1               = Column(Float)
    lag_7               = Column(Float)
    rolling_mean_7      = Column(Float)
    rolling_std_7       = Column(Float)
    predicted_demand    = Column(Float, nullable=False)
    safety_stock        = Column(Float)
    reorder_point       = Column(Float)
    reorder_quantity    = Column(Float)
    stockout_risk_pct   = Column(Float)
    health_score        = Column(Float)
    alert_code          = Column(String(20))
    alert_label         = Column(String(80))
    recommendations     = Column(Text)  # JSON serialised list

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} product={self.product_name!r} "
            f"demand={self.predicted_demand:.2f}>"
        )


class InventoryRecord(Base):
    """Daily inventory snapshots per product."""

    __tablename__ = "inventory"

    id                  = Column(Integer, primary_key=True, index=True)
    snapshot_date       = Column(DateTime, default=datetime.utcnow, index=True)
    product_name        = Column(String(120), index=True, nullable=False)
    category            = Column(String(80))
    current_stock       = Column(Float)
    avg_demand          = Column(Float)
    safety_stock        = Column(Float)
    reorder_point       = Column(Float)
    stockout_risk_pct   = Column(Float)
    health_score        = Column(Float)
    days_of_cover       = Column(Float)  # current_stock / avg_demand


class AlertRecord(Base):
    """Audit log of all inventory alerts raised."""

    __tablename__ = "alerts"

    id                  = Column(Integer, primary_key=True, index=True)
    raised_at           = Column(DateTime, default=datetime.utcnow, index=True)
    product_name        = Column(String(120), index=True)
    alert_code          = Column(String(20), index=True)   # STOCKOUT / OVERSTOCK / SPIKE / OK
    alert_label         = Column(String(80))
    predicted_demand    = Column(Float)
    current_stock       = Column(Float)
    reorder_point       = Column(Float)
    resolved            = Column(Boolean, default=False)
    resolved_at         = Column(DateTime, nullable=True)


class ForecastHistory(Base):
    """Batch forecast run results for trend analysis."""

    __tablename__ = "forecast_history"

    id                  = Column(Integer, primary_key=True, index=True)
    run_at              = Column(DateTime, default=datetime.utcnow, index=True)
    product_name        = Column(String(120), index=True)
    category            = Column(String(80))
    forecast_date       = Column(DateTime, index=True)
    predicted_demand    = Column(Float)
    actual_demand       = Column(Float, nullable=True)  # filled when actuals arrive
    mae                 = Column(Float, nullable=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def create_all_tables() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
