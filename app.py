"""
Streamlit Dashboard — Retail Demand Forecasting & Inventory Optimization
Clean, modern sales-operations dashboard powered by the trained model.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.business_logic import compute_inventory_analytics
from utils.ml_pipeline import PRODUCT_CATALOGUE, load_model, predict_demand

st.set_page_config(
    page_title="RetailIQ — Sales Ops Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0f172a;
        --panel: rgba(15, 23, 42, 0.88);
        --panel-border: rgba(148, 163, 184, 0.18);
        --text: #e2e8f0;
        --muted: #94a3b8;
        --accent: #6366f1;
        --accent-2: #38bdf8;
        --success: #4ade80;
        --warning: #f59e0b;
        --danger: #f87171;
        --shadow: 0 20px 45px rgba(15, 23, 42, 0.28);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(99, 102, 241, 0.22), transparent 18%),
            linear-gradient(135deg, #020617 0%, #0f172a 48%, #111827 100%);
    }

    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.95);
        border-right: 1px solid var(--panel-border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .hero {
        padding: 2rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
        color: #f8fafc;
    }

    .hero p {
        color: var(--muted);
        margin: 0;
    }

    .glass-card {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        padding: 1.1rem 1.2rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.4rem 0;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(99, 102, 241, 0.24);
        border-radius: 22px;
        padding: 1rem;
    }

    .metric-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.65rem 1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .pill-stockout { background: rgba(248, 113, 113, 0.12); color: #fecaca; border: 1px solid rgba(248, 113, 113, 0.3); }
    .pill-overstock { background: rgba(245, 158, 11, 0.12); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-spike { background: rgba(167, 139, 250, 0.12); color: #e9d5ff; border: 1px solid rgba(167, 139, 250, 0.3); }
    .pill-safe { background: rgba(74, 222, 128, 0.12); color: #bbf7d0; border: 1px solid rgba(74, 222, 128, 0.3); }

    .action-card {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        color: #e2e8f0;
        margin-bottom: 0.75rem;
    }

    .tiny-note {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.08rem;
        font-weight: 700;
        margin: 1.2rem 0 0.8rem 0;
    }

    .footer-note {
        color: #64748b;
        text-align: center;
        padding: 1rem 0;
        font-size: 0.82rem;
    }

    .upload-box {
        border: 1px dashed rgba(99, 102, 241, 0.5);
        border-radius: 20px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.8);
    }

    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(title: str, value: str, note: str, accent: str = "#6366f1") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value" style="color:{accent};">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def pill_class(alert_code: str) -> str:
    mapping = {
        "STOCKOUT": "pill-stockout",
        "OVERSTOCK": "pill-overstock",
        "SPIKE": "pill-spike",
        "OK": "pill-safe",
    }
    return mapping.get(alert_code, "pill-safe")


def normalize_uploaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")

    if "product_name" not in normalized.columns:
        normalized["product_name"] = [f"Product {index + 1}" for index in normalized.index]

    if "category" not in normalized.columns:
        if "category_Cold Drinks & Juices" in normalized.columns or "category_Snacks & Munchies" in normalized.columns:
            bev = normalized.get("category_Cold Drinks & Juices", 0).fillna(0)
            snack = normalized.get("category_Snacks & Munchies", 0).fillna(0)
            normalized["category"] = np.where(
                bev == 1,
                "Cold Drinks & Juices",
                np.where(snack == 1, "Snacks & Munchies", "Cold Drinks & Juices"),
            )
        else:
            normalized["category"] = "Cold Drinks & Juices"

    if "price" not in normalized.columns and "avg_unit_price" in normalized.columns:
        normalized["price"] = normalized["avg_unit_price"]
    elif "price" not in normalized.columns:
        normalized["price"] = 30.0

    if "promotion_flag" not in normalized.columns:
        normalized["promotion_flag"] = 0
    if "campaign_active" not in normalized.columns:
        normalized["campaign_active"] = 0
    if "marketing_intensity" not in normalized.columns and "total_spend" in normalized.columns:
        normalized["marketing_intensity"] = normalized["total_spend"]
    elif "marketing_intensity" not in normalized.columns:
        normalized["marketing_intensity"] = 0.5

    if "weekend_flag" not in normalized.columns:
        normalized["weekend_flag"] = 0
    if "festival_flag" not in normalized.columns:
        normalized["festival_flag"] = 0
    if "net_stock" not in normalized.columns:
        normalized["net_stock"] = 0.0

    for col in [
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
        "cross_product_demand_signal",
        "demand_ratio",
        "stock_turnover_rate",
        "quantity_sold",
        "net_stock",
        "price",
        "promotion_flag",
        "campaign_active",
        "weekend_flag",
        "festival_flag",
        "marketing_intensity",
    ]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "rolling_mean_7" not in normalized.columns and "rolling_7_mean" in normalized.columns:
        normalized["rolling_mean_7"] = normalized["rolling_7_mean"]
    if "rolling_std_7" not in normalized.columns and "rolling_7_std" in normalized.columns:
        normalized["rolling_std_7"] = normalized["rolling_7_std"]

    for col in ["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]:
        if col not in normalized.columns:
            normalized[col] = 0.0

    normalized["rolling_mean_7"] = normalized["rolling_mean_7"].fillna(normalized["rolling_mean_7"].median())
    normalized["rolling_std_7"] = normalized["rolling_std_7"].fillna(normalized["rolling_std_7"].median())
    normalized["lag_1"] = normalized["lag_1"].fillna(normalized["rolling_mean_7"])
    normalized["lag_7"] = normalized["lag_7"].fillna(normalized["rolling_mean_7"])
    normalized["price"] = normalized["price"].fillna(normalized["price"].median())
    normalized["net_stock"] = normalized["net_stock"].fillna(0.0)
    normalized["promotion_flag"] = normalized["promotion_flag"].fillna(0).astype(int)
    normalized["campaign_active"] = normalized["campaign_active"].fillna(0).astype(int)
    normalized["weekend_flag"] = normalized["weekend_flag"].fillna(0).astype(int)
    normalized["festival_flag"] = normalized["festival_flag"].fillna(0).astype(int)
    normalized["marketing_intensity"] = normalized["marketing_intensity"].fillna(0.5)

    if "cross_product_demand_signal" not in normalized.columns or normalized["cross_product_demand_signal"].isna().all():
        if "date" in normalized.columns and normalized["date"].notna().any():
            daily_cat = normalized.groupby(["date", "category"])["quantity_sold"].sum().reset_index()
            cat_pivot = daily_cat.pivot(index="date", columns="category", values="quantity_sold").reset_index().fillna(0)
            cat_pivot.columns.name = None
            if "Cold Drinks & Juices" in cat_pivot.columns and "Snacks & Munchies" in cat_pivot.columns:
                normalized = normalized.merge(
                    cat_pivot[["date", "Cold Drinks & Juices", "Snacks & Munchies"]],
                    on="date",
                    how="left",
                    suffixes=("", "_daily"),
                )
                normalized["cross_product_demand_signal"] = np.where(
                    normalized["category"] == "Snacks & Munchies",
                    normalized["Cold Drinks & Juices"],
                    normalized["Snacks & Munchies"],
                )
                normalized = normalized.drop(columns=["Cold Drinks & Juices", "Snacks & Munchies"], errors="ignore")
            else:
                normalized["cross_product_demand_signal"] = normalized["rolling_mean_7"]
        else:
            normalized["cross_product_demand_signal"] = normalized["rolling_mean_7"]

    normalized["cross_product_demand_signal"] = normalized["cross_product_demand_signal"].fillna(normalized["rolling_mean_7"])

    if "demand_ratio" not in normalized.columns or normalized["demand_ratio"].isna().all():
        if "date" in normalized.columns and normalized["date"].notna().any():
            daily_cat = normalized.groupby(["date", "category"])["quantity_sold"].sum().reset_index()
            bev_daily = daily_cat[daily_cat["category"] == "Cold Drinks & Juices"][["date", "quantity_sold"]].rename(columns={"quantity_sold": "beverage_demand_per_day"})
            snack_daily = daily_cat[daily_cat["category"] == "Snacks & Munchies"][["date", "quantity_sold"]].rename(columns={"quantity_sold": "snack_demand_per_day"})
            normalized = normalized.merge(bev_daily, on="date", how="left")
            normalized = normalized.merge(snack_daily, on="date", how="left")
            normalized["demand_ratio"] = normalized["beverage_demand_per_day"] / (normalized["snack_demand_per_day"] + 1)
            normalized = normalized.drop(columns=["beverage_demand_per_day", "snack_demand_per_day"], errors="ignore")
        else:
            normalized["demand_ratio"] = 1.0

    normalized["demand_ratio"] = normalized["demand_ratio"].fillna(1.0)
    normalized["stock_turnover_rate"] = (normalized["rolling_mean_7"] / normalized["net_stock"].clip(lower=1.0)).fillna(0.0)
    normalized["category_encoded"] = np.where(normalized["category"] == "Cold Drinks & Juices", 0, 1)
    normalized["total_spend"] = normalized["marketing_intensity"]
    normalized["promo_intensity"] = normalized["promotion_flag"] * normalized["marketing_intensity"]
    normalized["high_marketing_flag"] = (normalized["marketing_intensity"] > normalized["marketing_intensity"].median()).astype(int)

    return normalized


def build_prediction_input(row: pd.Series) -> Dict[str, Any]:
    return {
        "product_name": str(row.get("product_name", "Unknown Product")),
        "category": str(row.get("category", "Cold Drinks & Juices")),
        "price": float(row.get("price", 30.0)),
        "promotion_flag": int(row.get("promotion_flag", 0)),
        "campaign_active": int(row.get("campaign_active", 0)),
        "total_spend": float(row.get("total_spend", row.get("marketing_intensity", 0.5))),
        "net_stock": float(row.get("net_stock", 0.0)),
        "weekend_flag": int(row.get("weekend_flag", 0)),
        "festival_flag": int(row.get("festival_flag", 0)),
        "lag_1": float(row.get("lag_1", row.get("rolling_mean_7", 50.0))),
        "lag_7": float(row.get("lag_7", row.get("rolling_mean_7", 50.0))),
        "rolling_mean_7": float(row.get("rolling_mean_7", 50.0)),
        "rolling_std_7": float(row.get("rolling_std_7", 5.0)),
        "cross_product_demand_signal": float(row.get("cross_product_demand_signal", row.get("rolling_mean_7", 50.0))),
        "demand_ratio": float(row.get("demand_ratio", 1.0)),
    }


def build_uploaded_summary(df: pd.DataFrame, model) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        prediction_input = build_prediction_input(row)
        pred_result = predict_demand(prediction_input, model=model)
        analytics = compute_inventory_analytics(
            product_name=prediction_input["product_name"],
            predicted_demand=pred_result["predicted_demand"],
            current_stock=prediction_input["net_stock"],
            rolling_mean_7=prediction_input["rolling_mean_7"],
            rolling_std_7=prediction_input["rolling_std_7"],
            weekend_flag=prediction_input["weekend_flag"],
            festival_flag=prediction_input["festival_flag"],
            promotion_flag=prediction_input["promotion_flag"],
        )
        rows.append(
            {
                "product_name": prediction_input["product_name"],
                "category": prediction_input["category"],
                "current_stock": prediction_input["net_stock"],
                "predicted_demand": pred_result["predicted_demand"],
                "reorder_quantity": analytics["reorder_quantity"],
                "stockout_risk_pct": analytics["stockout_risk_pct"],
                "health_score": analytics["health_score"],
                "alert_code": analytics["alert_code"],
                "alert_label": analytics["alert_label"],
                "recommendations": analytics["recommendations"],
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    return summary_df.sort_values(by=["stockout_risk_pct", "health_score"], ascending=[False, True]).reset_index(drop=True)


with st.sidebar:
    st.markdown("### 🧭 Mode")
    mode = st.radio("Choose dashboard mode", ["Single SKU", "Upload CSV"], horizontal=False)

    if mode == "Single SKU":
        st.markdown("### 🏷️ Product")
        product_name = st.selectbox("Select Product", list(PRODUCT_CATALOGUE.keys()))
        prod_info = PRODUCT_CATALOGUE[product_name]
        category = prod_info["category"]
        st.caption(f"Category: {category}")

        st.markdown("### 💰 Pricing")
        price = st.slider("Unit Price (₹)", 5.0, 200.0, float(prod_info["price"]), step=5.0)

        st.markdown("### 📊 Demand Context")
        lag_1 = st.number_input("Lag-1 Demand", min_value=0.0, value=55.0, step=1.0)
        lag_7 = st.number_input("Lag-7 Demand", min_value=0.0, value=52.0, step=1.0)
        rolling_mean_7 = st.number_input("Rolling Mean-7", min_value=0.0, value=53.0, step=1.0)
        rolling_std_7 = st.number_input("Rolling Std-7", min_value=0.0, value=5.0, step=0.5)

        st.markdown("### 📦 Stock")
        current_stock = st.number_input("Current Stock", min_value=0.0, value=80.0, step=5.0)

        col_a, col_b = st.columns(2)
        promotion_flag = int(col_a.checkbox("Promotion", value=False))
        campaign_active = int(col_b.checkbox("Campaign", value=False))

        col_c, col_d = st.columns(2)
        festival_flag = int(col_c.checkbox("Festival", value=False))
        weekend_flag = int(col_d.checkbox("Weekend", value=False))

        st.markdown("### 📈 Marketing")
        total_spend = st.slider("Marketing Intensity", 0.0, 1.0, 0.5, step=0.05)

        st.markdown("### 🔗 Cross-category")
        cross_signal = st.number_input("Opposite Category Demand", min_value=0.0, value=50.0, step=5.0)
        demand_ratio = st.slider("Bev/Snack Demand Ratio", 0.1, 5.0, 1.0, step=0.05)

        st.divider()
        run_prediction = st.button("Run forecast", type="primary", width="stretch")
    else:
        st.markdown("### ⬆️ Upload store CSV")
        st.caption("Use your own file to populate the dashboard. The app will run the trained model against each row.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        st.caption("Recommended columns: product_name, date, category (or category_*), price / avg_unit_price, net_stock, promotion_flag, campaign_active, marketing_intensity, lag_1, lag_7, rolling_7_mean / rolling_mean_7, rolling_7_std / rolling_std_7, quantity_sold.")

        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            normalized_df = normalize_uploaded_frame(raw_df)
            st.session_state["uploaded_df"] = normalized_df
            st.success(f"Loaded {len(normalized_df)} rows from {uploaded_file.name}.")

st.markdown(
    """
    <div class="hero">
        <h1>RetailIQ Sales Ops Dashboard</h1>
        <p>Clean, practical forecast & replenishment view for your store teams. No charts, just the actions that matter.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

model = None
model_status = "ready"
try:
    model = load_model()
except FileNotFoundError:
    model_status = "missing"

if mode == "Single SKU":
    input_data = {
        "product_name": product_name,
        "category": category,
        "price": price,
        "promotion_flag": promotion_flag,
        "campaign_active": campaign_active,
        "total_spend": total_spend,
        "net_stock": current_stock,
        "weekend_flag": weekend_flag,
        "festival_flag": festival_flag,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7,
        "cross_product_demand_signal": cross_signal,
        "demand_ratio": demand_ratio,
    }

    if run_prediction or "single_result" not in st.session_state:
        with st.spinner("Running trained model…"):
            if model is not None:
                pred_result = predict_demand(input_data, model=model)
                pred_demand = pred_result["predicted_demand"]
                analytics = compute_inventory_analytics(
                    product_name=product_name,
                    predicted_demand=pred_demand,
                    current_stock=current_stock,
                    rolling_mean_7=rolling_mean_7,
                    rolling_std_7=rolling_std_7,
                    weekend_flag=weekend_flag,
                    festival_flag=festival_flag,
                    promotion_flag=promotion_flag,
                )
            else:
                pred_demand = rolling_mean_7
                analytics = compute_inventory_analytics(
                    product_name=product_name,
                    predicted_demand=pred_demand,
                    current_stock=current_stock,
                    rolling_mean_7=rolling_mean_7,
                    rolling_std_7=rolling_std_7,
                    weekend_flag=weekend_flag,
                    festival_flag=festival_flag,
                    promotion_flag=promotion_flag,
                )
            st.session_state["single_result"] = {"pred_demand": pred_demand, "analytics": analytics}
    else:
        pred_demand = st.session_state["single_result"]["pred_demand"]
        analytics = st.session_state["single_result"]["analytics"]

    if model_status == "missing":
        st.warning("Model file is missing. Showing fallback calculations from the current inputs.", icon="⚠️")

    st.markdown("<div class='section-title'>At a glance</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='glass-card'>
            <div style='display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center'>
                <div>
                    <div style='font-size:0.9rem; color:#94a3b8;'>Current SKU</div>
                    <div style='font-size:1.3rem; font-weight:800; color:#f8fafc;'>{product_name}</div>
                    <div class='tiny-note'>{category}</div>
                </div>
                <div>
                    <span class='status-pill {pill_class(analytics['alert_code'])}'>{analytics['alert_label']}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    st.markdown(metric_card("Predicted Demand", f"{pred_demand:.1f}", "units for today", "#6366f1"), unsafe_allow_html=True)
    st.markdown(metric_card("Stockout Risk", f"{analytics['stockout_risk_pct']:.0f}%", "fresh risk score", "#f87171"), unsafe_allow_html=True)
    st.markdown(metric_card("Reorder Qty", f"{analytics['reorder_quantity']:.0f}", "units to order", "#fb923c"), unsafe_allow_html=True)
    st.markdown(metric_card("Safety Stock", f"{analytics['safety_stock']:.0f}", "buffer units", "#38bdf8"), unsafe_allow_html=True)
    st.markdown(metric_card("Health Score", f"{analytics['health_score']:.0f}/100", "inventory health", "#4ade80"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Recommended actions</div>", unsafe_allow_html=True)
    for rec in analytics["recommendations"]:
        st.markdown(f"<div class='action-card'>{rec}</div>", unsafe_allow_html=True)

    with st.expander("📋 Full metrics", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": [
                            "Product",
                            "Category",
                            "Predicted Demand",
                            "Safety Stock",
                            "Reorder Point",
                            "Reorder Quantity",
                            "Stockout Risk",
                            "Health Score",
                            "Alert",
                            "Current Stock",
                        ],
                        "Value": [
                            product_name,
                            category,
                            f"{pred_demand:.2f} units",
                            f"{analytics['safety_stock']:.2f} units",
                            f"{analytics['reorder_point']:.2f} units",
                            f"{analytics['reorder_quantity']:.2f} units",
                            f"{analytics['stockout_risk_pct']:.1f}%",
                            f"{analytics['health_score']:.1f}/100",
                            analytics["alert_label"],
                            f"{current_stock:.0f} units",
                        ],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
else:
    if "uploaded_df" not in st.session_state:
        st.info("Upload a CSV file to begin using your live store data.")
        st.stop()

    uploaded_df = st.session_state["uploaded_df"]
    if model is None:
        st.error("The trained model is not available. Run `python train_model.py --data your_data.csv` first.")
        st.stop()

    with st.spinner("Running your store data through the trained model…"):
        summary_df = build_uploaded_summary(uploaded_df, model)

    if summary_df.empty:
        st.warning("The uploaded file does not contain enough usable rows to generate predictions.")
        st.stop()

    total_rows = len(summary_df)
    total_demand = float(summary_df["predicted_demand"].sum())
    total_reorder = float(summary_df["reorder_quantity"].sum())
    urgent_count = int((summary_df["alert_code"] != "OK").sum())
    safe_count = int((summary_df["alert_code"] == "OK").sum())
    avg_health = float(summary_df["health_score"].mean())

    st.markdown("<div class='section-title'>Store-wide snapshot</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    st.markdown(metric_card("Rows processed", f"{total_rows}", "uploaded records", "#38bdf8"), unsafe_allow_html=True)
    st.markdown(metric_card("Predicted demand", f"{total_demand:.0f}", "units across the file", "#6366f1"), unsafe_allow_html=True)
    st.markdown(metric_card("Reorder volume", f"{total_reorder:.0f}", "units to replenish", "#fb923c"), unsafe_allow_html=True)
    st.markdown(metric_card("Urgent SKUs", f"{urgent_count}", "need immediate action", "#f87171"), unsafe_allow_html=True)
    st.markdown(metric_card("Average health", f"{avg_health:.0f}/100", "inventory health", "#4ade80"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Top action items</div>", unsafe_allow_html=True)
    critical = summary_df[summary_df["alert_code"] != "OK"].head(5)
    if critical.empty:
        st.success("No urgent items found in the uploaded file.")
    else:
        for _, row in critical.iterrows():
            st.markdown(
                f"<div class='action-card'><strong>{row['product_name']}</strong> — {row['alert_label']}<br><span class='tiny-note'>Predicted demand: {row['predicted_demand']:.1f} units · Reorder: {row['reorder_quantity']:.0f} · Risk: {row['stockout_risk_pct']:.0f}%</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Priority table</div>", unsafe_allow_html=True)
    st.dataframe(
        summary_df[
            [
                "product_name",
                "category",
                "current_stock",
                "predicted_demand",
                "reorder_quantity",
                "stockout_risk_pct",
                "health_score",
                "alert_label",
            ]
        ].rename(
            columns={
                "product_name": "Product",
                "category": "Category",
                "current_stock": "Current Stock",
                "predicted_demand": "Predicted Demand",
                "reorder_quantity": "Reorder Qty",
                "stockout_risk_pct": "Risk %",
                "health_score": "Health",
                "alert_label": "Status",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("<div class='section-title'>Recommendations</div>", unsafe_allow_html=True)
    for _, row in summary_df.head(8).iterrows():
        for rec in row["recommendations"][:2]:
            st.markdown(f"<div class='action-card'>{rec}</div>", unsafe_allow_html=True)

    report = {
        "source": uploaded_file.name,
        "rows_processed": int(total_rows),
        "total_predicted_demand": round(total_demand, 2),
        "total_reorder_qty": round(total_reorder, 2),
        "urgent_items": int(urgent_count),
        "safe_items": int(safe_count),
        "summary": summary_df.to_dict(orient="records"),
    }

    with st.expander("💾 Export report", expanded=False):
        st.download_button(
            label="Download JSON report",
            data=json.dumps(report, indent=2),
            file_name="retailiq_store_upload_report.json",
            mime="application/json",
            width="stretch",
        )

st.markdown("<div class='footer-note'>RetailIQ v1.0 — trained model + live store input · clean sales ops view.</div>", unsafe_allow_html=True)
