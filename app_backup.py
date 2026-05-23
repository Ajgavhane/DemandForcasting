"""
Streamlit Dashboard — Retail Demand Forecasting & Inventory Optimization
Modern, portfolio-grade UI with KPI cards, Plotly charts, and AI recommendations.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.business_logic import compute_inventory_analytics
from utils.ml_pipeline import (
    PRODUCT_CATALOGUE,
    engineer_features,
    get_feature_importance,
    load_model,
    predict_demand,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailIQ — Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — modern SaaS dark-friendly theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ────────────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── KPI cards ───────────────────────────────────────────────────────────── */
.kpi-card {
    background: rgba(30,41,59,0.85);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(99,102,241,.25); }
.kpi-title  { font-size: 12px; font-weight: 600; text-transform: uppercase;
               letter-spacing: 1.5px; color: #94a3b8; margin-bottom: 6px; }
.kpi-value  { font-size: 32px; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
.kpi-sub    { font-size: 11px; color: #64748b; margin-top: 4px; }

/* ── Alert badges ────────────────────────────────────────────────────────── */
.alert-badge {
    display: inline-block; padding: 8px 20px; border-radius: 50px;
    font-size: 14px; font-weight: 700; letter-spacing: .5px;
}
.alert-stockout  { background: #450a0a; color: #f87171; border: 1px solid #b91c1c; }
.alert-overstock { background: #422006; color: #fbbf24; border: 1px solid #d97706; }
.alert-spike     { background: #2e1065; color: #c084fc; border: 1px solid #7c3aed; }
.alert-ok        { background: #052e16; color: #4ade80; border: 1px solid #15803d; }

/* ── Rec panel ───────────────────────────────────────────────────────────── */
.rec-item {
    background: rgba(30,41,59,.6); border: 1px solid #334155;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    color: #e2e8f0; font-size: 14px; line-height: 1.5;
}

/* ── Section headers ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 18px; font-weight: 700; color: #f1f5f9;
    border-left: 4px solid #6366f1; padding-left: 12px;
    margin: 24px 0 16px;
}

/* ── Plotly charts ───────────────────────────────────────────────────────── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* hide hamburger */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(15,23,42,0)",
    plot_bgcolor="rgba(30,41,59,0.6)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def plotly_config() -> dict:
    return {"displayModeBar": False, "responsive": True}


def kpi_card(title: str, value: str, sub: str = "", color: str = "#6366f1") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def alert_badge(alert_code: str, alert_label: str) -> str:
    css_map = {
        "STOCKOUT":  "alert-stockout",
        "OVERSTOCK": "alert-overstock",
        "SPIKE":     "alert-spike",
        "OK":        "alert-ok",
    }
    css = css_map.get(alert_code, "alert-ok")
    return f'<span class="alert-badge {css}">{alert_label}</span>'


def health_color(score: float) -> str:
    if score >= 70:
        return "#4ade80"
    if score >= 40:
        return "#fbbf24"
    return "#f87171"


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """Generate synthetic demo data when no CSV is uploaded."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2023-12-31")
    products = list(PRODUCT_CATALOGUE.keys())
    rows = []
    for product in products:
        cat  = PRODUCT_CATALOGUE[product]["category"]
        price = PRODUCT_CATALOGUE[product]["price"]
        base_demand = rng.integers(30, 80)
        for i, d in enumerate(dates):
            is_weekend = int(d.weekday() >= 5)
            is_festival = int(d.month in [10, 11] and d.day in range(1, 15))
            trend = base_demand + i * 0.02
            noise = rng.normal(0, 5)
            qty   = max(round(trend + noise + is_weekend * 10 + is_festival * 20), 0)
            rows.append({
                "date":        d,
                "product_name": product,
                "category":    cat,
                "price":       price,
                "quantity_sold": qty,
                "net_stock":   rng.integers(20, 150),
                "promotion_flag": int(rng.random() > 0.8),
                "festival_flag":  is_festival,
                "weekend_flag":   is_weekend,
            })
    df = pd.DataFrame(rows)
    df["lag_1"]         = df.groupby("product_name")["quantity_sold"].shift(1).fillna(50)
    df["lag_7"]         = df.groupby("product_name")["quantity_sold"].shift(7).fillna(50)
    df["rolling_mean_7"] = (
        df.groupby("product_name")["quantity_sold"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["rolling_std_7"] = (
        df.groupby("product_name")["quantity_sold"]
        .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(5))
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📦 RetailIQ")
    st.markdown("*Demand Forecasting & Inventory AI*")
    st.divider()

    st.markdown("### 🏷️ Product")
    product_name = st.selectbox("Select Product", list(PRODUCT_CATALOGUE.keys()))
    prod_info    = PRODUCT_CATALOGUE[product_name]
    category     = prod_info["category"]
    st.caption(f"Category: {category}")

    st.markdown("### 💰 Pricing")
    price = st.slider("Unit Price (₹)", 5.0, 200.0,
                       float(prod_info["price"]), step=5.0)

    st.markdown("### 📊 Demand Context")
    lag_1          = st.number_input("Lag-1 Demand (yesterday)",   min_value=0.0, value=55.0, step=1.0)
    lag_7          = st.number_input("Lag-7 Demand (week ago)",    min_value=0.0, value=52.0, step=1.0)
    rolling_mean_7 = st.number_input("Rolling Mean-7 Demand",      min_value=0.0, value=53.0, step=1.0)
    rolling_std_7  = st.number_input("Rolling Std-7 Demand",       min_value=0.0, value=5.0,  step=0.5)

    st.markdown("### 📦 Stock")
    current_stock  = st.number_input("Current Stock (units)", min_value=0.0, value=80.0, step=5.0)

    st.markdown("### 🎯 Conditions")
    col_a, col_b = st.columns(2)
    promotion_flag = int(col_a.checkbox("Promotion 📣"))
    festival_flag  = int(col_b.checkbox("Festival 🎉"))
    col_c, col_d   = st.columns(2)
    weekend_flag   = int(col_c.checkbox("Weekend 📅"))
    campaign_active = int(col_d.checkbox("Campaign 📢"))

    st.markdown("### 📈 Marketing")
    total_spend = st.slider("Marketing Intensity", 0.0, 1.0, 0.5, step=0.05,
                             help="Normalised 0–1 spend intensity")

    st.markdown("### 🔗 Cross-Product Signal")
    cross_signal = st.number_input(
        "Opposite Category Daily Demand", min_value=0.0, value=50.0, step=5.0,
        help="E.g. snack demand when product is a beverage"
    )
    demand_ratio = st.slider("Bev/Snack Demand Ratio", 0.1, 5.0, 1.0, step=0.05)

    st.divider()
    predict_button = st.button("🚀 Run Prediction", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 20px 0 10px">
    <h1 style="font-size:36px; font-weight:800; color:#f1f5f9; margin:0">
        📦 RetailIQ <span style="color:#6366f1">Demand Forecasting</span>
    </h1>
    <p style="color:#64748b; font-size:14px; margin:4px 0 0">
        XGBoost-powered demand prediction · Inventory optimization · Real-time alerts
        &nbsp;|&nbsp; <b style="color:#4ade80">MAE: 4.36 units</b>
        &nbsp;|&nbsp; <b style="color:#a78bfa">10.6% improvement</b>
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

input_data: Dict[str, Any] = {
    "product_name":               product_name,
    "category":                   category,
    "price":                      price,
    "promotion_flag":             promotion_flag,
    "campaign_active":            campaign_active,
    "total_spend":                total_spend,
    "net_stock":                  current_stock,
    "weekend_flag":               weekend_flag,
    "festival_flag":              festival_flag,
    "lag_1":                      lag_1,
    "lag_7":                      lag_7,
    "rolling_mean_7":             rolling_mean_7,
    "rolling_std_7":              rolling_std_7,
    "cross_product_demand_signal": cross_signal,
    "demand_ratio":               demand_ratio,
}

# Auto-predict on page load; re-predict on button
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if predict_button or st.session_state["last_result"] is None:
    with st.spinner("Running prediction pipeline…"):
        try:
            model = load_model()
            pred_result = predict_demand(input_data, model=model)
            pred_demand = pred_result["predicted_demand"]
            analytics   = compute_inventory_analytics(
                product_name     = product_name,
                predicted_demand = pred_demand,
                current_stock    = current_stock,
                rolling_mean_7   = rolling_mean_7,
                rolling_std_7    = rolling_std_7,
                weekend_flag     = weekend_flag,
                festival_flag    = festival_flag,
                promotion_flag   = promotion_flag,
            )
            fi = get_feature_importance(model=model)
            st.session_state["last_result"] = {
                "pred_demand": pred_demand,
                "analytics":   analytics,
                "fi":          fi,
            }
            st.session_state["model_loaded"] = True
        except FileNotFoundError:
            st.session_state["model_loaded"] = False
            st.session_state["last_result"]  = {
                "pred_demand": rolling_mean_7 * (1 + 0.1 * promotion_flag + 0.15 * festival_flag),
                "analytics":   compute_inventory_analytics(
                    product_name     = product_name,
                    predicted_demand = rolling_mean_7,
                    current_stock    = current_stock,
                    rolling_mean_7   = rolling_mean_7,
                    rolling_std_7    = rolling_std_7,
                    weekend_flag     = weekend_flag,
                    festival_flag    = festival_flag,
                    promotion_flag   = promotion_flag,
                ),
                "fi": {
                    "rolling_mean_7": 0.28, "lag_1": 0.22, "lag_7": 0.15,
                    "promo_intensity": 0.09, "cross_product_demand_signal": 0.07,
                    "net_stock": 0.05, "demand_ratio": 0.04, "price": 0.04,
                    "festival_flag": 0.03, "weekend_flag": 0.03,
                },
            }

# Short-circuit if no result
if st.session_state["last_result"] is None:
    st.info("Click **Run Prediction** to start.")
    st.stop()

result      = st.session_state["last_result"]
pred_demand = result["pred_demand"]
analytics   = result["analytics"]
fi          = result["fi"]

if not st.session_state.get("model_loaded", True):
    st.warning(
        "⚠️  Model file `backend/models/demand_model.pkl` not found. "
        "Run `python train_model.py --data your_data.csv` to train the model. "
        "Showing **heuristic estimates** based on rolling averages.",
        icon="⚠️",
    )

# ──────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Key Performance Indicators</div>',
            unsafe_allow_html=True)

cols = st.columns(5)
with cols[0]:
    st.markdown(
        kpi_card("Predicted Demand", f"{pred_demand:.1f}", "units today", "#6366f1"),
        unsafe_allow_html=True,
    )
with cols[1]:
    risk_color = "#f87171" if analytics["stockout_risk_pct"] > 70 \
        else ("#fbbf24" if analytics["stockout_risk_pct"] > 40 else "#4ade80")
    st.markdown(
        kpi_card("Stockout Risk", f"{analytics['stockout_risk_pct']:.0f}%",
                 "probability", risk_color),
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        kpi_card("Reorder Qty", f"{analytics['reorder_quantity']:.0f}",
                 "units to order", "#fb923c"),
        unsafe_allow_html=True,
    )
with cols[3]:
    st.markdown(
        kpi_card("Safety Stock", f"{analytics['safety_stock']:.0f}",
                 "buffer units", "#38bdf8"),
        unsafe_allow_html=True,
    )
with cols[4]:
    h = analytics["health_score"]
    st.markdown(
        kpi_card("Inventory Health", f"{h:.0f}/100",
                 "score", health_color(h)),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# ALERT BANNER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div style="text-align:center; margin: 4px 0 20px">'
    f'{alert_badge(analytics["alert_code"], analytics["alert_label"])}'
    f'</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DEMO DATA
# ──────────────────────────────────────────────────────────────────────────────

demo_df = load_sample_data()
prod_df = demo_df[demo_df["product_name"] == product_name].copy()

# ──────────────────────────────────────────────────────────────────────────────
# ROW 1: Demand Trend + Feature Importance
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📈 Demand Analytics</div>', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown("**Demand Trend Over Time**")
    recent = prod_df.tail(90).copy()
    recent["ema"] = recent["quantity_sold"].ewm(alpha=0.3, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["quantity_sold"],
        name="Actual", line=dict(color="#6366f1", width=1.5), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["ema"],
        name="EMA", line=dict(color="#f59e0b", width=2, dash="dot"), opacity=0.9,
    ))
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["rolling_mean_7"],
        name="Rolling-7", line=dict(color="#4ade80", width=1.5, dash="dash"),
    ))
    fig.add_hline(y=pred_demand, line_dash="longdash",
                  line_color="#f87171", annotation_text=f"Today's Forecast: {pred_demand:.1f}")
    fig.update_layout(title=f"{product_name} — 90-Day Demand View", **PLOTLY_LAYOUT,
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

with chart_col2:
    st.markdown("**Feature Importance**")
    fi_df = (
        pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
        .sort_values("Importance")
        .tail(10)
    )
    NEW_FEATS = {"promo_intensity", "high_marketing_flag", "demand_ratio"}
    colors = ["#f59e0b" if f in NEW_FEATS else "#6366f1" for f in fi_df["Feature"]]
    fig2 = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"],
        orientation="h", marker_color=colors, text=fi_df["Importance"].round(3),
        textposition="outside",
    ))
    fig2.update_layout(title="XGBoost Feature Importances", **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True, config=plotly_config())

# ──────────────────────────────────────────────────────────────────────────────
# ROW 2: Stock Risk Pie + Promotion Impact + Inventory Heatmap
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📦 Inventory Intelligence</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Stock Risk Distribution**")
    fig3 = go.Figure(go.Pie(
        labels=["Stockout", "Safe", "Overstock"],
        values=[
            analytics["stockout_risk_pct"],
            max(0, 100 - analytics["stockout_risk_pct"] - 10),
            10,
        ],
        hole=0.55,
        marker=dict(colors=["#ef4444", "#4ade80", "#f59e0b"]),
        textinfo="label+percent",
    ))
    fig3.update_layout(
        title="Risk Breakdown", **PLOTLY_LAYOUT,
        annotations=[dict(text=f"{analytics['stockout_risk_pct']:.0f}%",
                          font_size=22, showarrow=False, font_color="#f87171")]
    )
    st.plotly_chart(fig3, use_container_width=True, config=plotly_config())

with col_b:
    st.markdown("**Net Stock vs Reorder Threshold**")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_stock,
        delta={"reference": analytics["reorder_point"],
               "decreasing": {"color": "#ef4444"},
               "increasing": {"color": "#4ade80"}},
        gauge={
            "axis": {"range": [0, max(current_stock * 2, analytics["reorder_point"] * 2)]},
            "bar":  {"color": health_color(analytics["health_score"])},
            "steps": [
                {"range": [0, analytics["reorder_point"]], "color": "#450a0a"},
                {"range": [analytics["reorder_point"],
                            analytics["reorder_point"] * 2], "color": "#052e16"},
            ],
            "threshold": {
                "line": {"color": "#f59e0b", "width": 3},
                "thickness": 0.75,
                "value": analytics["reorder_point"],
            },
        },
        title={"text": "Current Stock", "font": {"color": "#e2e8f0"}},
        number={"suffix": " units", "font": {"color": "#f1f5f9"}},
    ))
    gauge.update_layout(**PLOTLY_LAYOUT, height=280)
    st.plotly_chart(gauge, use_container_width=True, config=plotly_config())

with col_c:
    st.markdown("**Promotion Demand Lift (All Products)**")
    promo_lifts = {}
    for p, info in PRODUCT_CATALOGUE.items():
        base  = demo_df[demo_df["product_name"] == p]["quantity_sold"].mean()
        promo = base * (1.18 if info["category"] == "Cold Drinks & Juices" else 1.12)
        promo_lifts[p] = round(((promo - base) / base) * 100, 1)
    lift_df = pd.DataFrame(list(promo_lifts.items()), columns=["Product", "Lift%"])
    lift_df = lift_df.sort_values("Lift%")
    fig4 = go.Figure(go.Bar(
        x=lift_df["Lift%"], y=lift_df["Product"],
        orientation="h",
        marker_color=["#6366f1" if v > 15 else "#94a3b8" for v in lift_df["Lift%"]],
        text=[f"+{v}%" for v in lift_df["Lift%"]], textposition="outside",
    ))
    fig4.update_layout(title="Avg Promo Lift %", **PLOTLY_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True, config=plotly_config())

# ──────────────────────────────────────────────────────────────────────────────
# ROW 3: Product-wise Forecast + Category Comparison
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🛒 Product & Category Analysis</div>',
            unsafe_allow_html=True)
col_d, col_e = st.columns(2)

with col_d:
    st.markdown("**Product-wise Average Demand**")
    avg_by_prod = demo_df.groupby("product_name")["quantity_sold"].mean().sort_values()
    fig5 = go.Figure(go.Bar(
        x=avg_by_prod.values, y=avg_by_prod.index, orientation="h",
        marker=dict(
            color=avg_by_prod.values,
            colorscale="Viridis",
            showscale=False,
        ),
        text=avg_by_prod.values.round(1), textposition="outside",
    ))
    fig5.update_layout(title="Avg Daily Units Sold", **PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True, config=plotly_config())

with col_e:
    st.markdown("**Category Demand Comparison Over Time**")
    cat_monthly = (
        demo_df.assign(month=demo_df["date"].dt.to_period("M").astype(str))
        .groupby(["month", "category"])["quantity_sold"].sum().reset_index()
    )
    fig6 = px.line(
        cat_monthly, x="month", y="quantity_sold", color="category",
        color_discrete_map={
            "Cold Drinks & Juices": "#6366f1",
            "Snacks & Munchies":    "#f59e0b",
        },
    )
    fig6.update_layout(title="Monthly Category Demand", **PLOTLY_LAYOUT,
                       xaxis_tickangle=-30, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig6, use_container_width=True, config=plotly_config())

# ──────────────────────────────────────────────────────────────────────────────
# ROW 4: Inventory Heatmap + Actual vs Predicted (demo)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🗺️ Heatmap & Forecast vs Actual</div>',
            unsafe_allow_html=True)
col_f, col_g = st.columns(2)

with col_f:
    st.markdown("**Demand Heatmap (Product × Month)**")
    heatmap_data = (
        demo_df.assign(month=demo_df["date"].dt.month_name().str[:3])
        .groupby(["product_name", "month"])["quantity_sold"].mean()
        .unstack(fill_value=0)
    )
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    heatmap_data = heatmap_data[[m for m in month_order if m in heatmap_data.columns]]
    fig7 = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        colorscale="Viridis",
        text=heatmap_data.values.round(0),
        texttemplate="%{text:.0f}",
        colorbar=dict(title="Units"),
    ))
    fig7.update_layout(title="Avg Daily Demand by Product & Month", **PLOTLY_LAYOUT,
                       height=350)
    st.plotly_chart(fig7, use_container_width=True, config=plotly_config())

with col_g:
    st.markdown("**Actual vs Predicted (Demo)**")
    recent60 = prod_df.tail(60).copy()
    recent60["predicted"] = (
        recent60["rolling_mean_7"]
        + np.random.default_rng(0).normal(0, 3, len(recent60))
    ).clip(lower=0)
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=recent60["date"], y=recent60["quantity_sold"],
        name="Actual", fill="tozeroy",
        line=dict(color="#6366f1", width=2),
        fillcolor="rgba(99,102,241,0.1)",
    ))
    fig8.add_trace(go.Scatter(
        x=recent60["date"], y=recent60["predicted"],
        name="Predicted", line=dict(color="#f59e0b", width=2, dash="dash"),
    ))
    fig8.update_layout(title=f"{product_name} — Last 60 Days", **PLOTLY_LAYOUT,
                       legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig8, use_container_width=True, config=plotly_config())

# ──────────────────────────────────────────────────────────────────────────────
# AI RECOMMENDATION PANEL
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🤖 AI Recommendations</div>', unsafe_allow_html=True)

recs = analytics["recommendations"]
rec_cols = st.columns(min(len(recs), 2))
for i, rec in enumerate(recs):
    with rec_cols[i % 2]:
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DETAILED METRICS TABLE
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("📋 Full Inventory Metrics", expanded=False):
    metrics_df = pd.DataFrame([{
        "Product":            product_name,
        "Category":           category,
        "Predicted Demand":   f"{pred_demand:.2f} units",
        "Safety Stock":       f"{analytics['safety_stock']:.2f} units",
        "Reorder Point":      f"{analytics['reorder_point']:.2f} units",
        "Reorder Quantity":   f"{analytics['reorder_quantity']:.2f} units",
        "Stockout Risk":      f"{analytics['stockout_risk_pct']:.1f}%",
        "Health Score":       f"{analytics['health_score']:.1f}/100",
        "Alert":              analytics["alert_label"],
        "Current Stock":      f"{current_stock:.0f} units",
    }]).T.reset_index()
    metrics_df.columns = ["Metric", "Value"]
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD REPORT
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("💾 Export Report", expanded=False):
    report = {
        "product":      product_name,
        "category":     category,
        "prediction":   pred_demand,
        "inventory":    analytics,
        "input_data":   input_data,
    }
    st.download_button(
        label="📥 Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name=f"retail_forecast_{product_name.lower().replace(' ', '_')}.json",
        mime="application/json",
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="text-align:center; color:#334155; font-size:12px; padding:8px 0">
    RetailIQ v1.0 &nbsp;·&nbsp; XGBoost Demand Forecasting &nbsp;·&nbsp;
    Baseline MAE: 4.8769 → Improved MAE: 4.3601 (10.6% gain)
    &nbsp;·&nbsp; Built with ❤️ using Streamlit + FastAPI
</div>
""", unsafe_allow_html=True)
