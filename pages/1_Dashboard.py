import streamlit as st
import pandas as pd
import joblib
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/shipment_delay_xgb_model.pkl")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/shipment_clean_data.csv")

# -------------------------------
# Load Project Objects
# -------------------------------
@st.cache_resource
def load_objects():
    with open("models/project_objects.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
df = load_data()
objects = load_objects()
# ---------------------------------------
# Sidebar Filters
# ---------------------------------------

st.sidebar.header("🔎 Filters")

selected_market = st.sidebar.selectbox(
    "Market",
    ["All"] + sorted(df["Market"].unique().tolist())
)

selected_segment = st.sidebar.selectbox(
    "Customer Segment",
    ["All"] + sorted(df["Customer Segment"].unique().tolist())
)

st.title("📊 Executive Dashboard")
st.markdown("---")
# ---------------------------------------
# Apply Filters
# ---------------------------------------

filtered_df = df.copy()

if selected_market != "All":
    filtered_df = filtered_df[
        filtered_df["Market"] == selected_market
    ]

if selected_segment != "All":
    filtered_df = filtered_df[
        filtered_df["Customer Segment"] == selected_segment
    ]
# ---------------------------------------
# KPI Calculations
# ---------------------------------------

total_shipments = len(filtered_df)

delayed_shipments = filtered_df["Late_delivery_risk"].sum()

delay_rate = (delayed_shipments / total_shipments) * 100

# Check if Risk_Level exists
if "Risk_Level" in filtered_df.columns:
    high_risk = (filtered_df["Risk_Level"] == "High Risk").sum()
else:
    high_risk = 0

model_accuracy = 78.9
roc_auc = 0.84
# ---------------------------------------
# KPI Cards
# ---------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📦 Total Shipments",
        f"{total_shipments:,}"
    )

with col2:
    st.metric(
        "⏰ Delayed Shipments",
        f"{delayed_shipments:,}"
    )

with col3:
    st.metric(
        "📈 Delay Rate",
        f"{delay_rate:.2f}%"
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "⚠ High Risk Shipments",
        f"{high_risk:,}"
    )

with col5:
    st.metric(
        "🎯 Model Accuracy",
        f"{model_accuracy}%"
    )

with col6:
    st.metric(
        "📊 ROC-AUC Score",
        roc_auc
    )
# ---------------------------------------
# Shipment Distribution by Market
# ---------------------------------------

# ---------------------------------------
# Delay Rate by Market
# ---------------------------------------

st.markdown("---")

st.subheader("🌍 Delay Rate by Market")

market_delay = (
    filtered_df
    .groupby("Market")["Late_delivery_risk"]
    .mean()
    .reset_index()
)

market_delay["Delay Rate (%)"] = market_delay["Late_delivery_risk"] * 100

fig = px.bar(
    market_delay,
    x="Market",
    y="Delay Rate (%)",
    text="Delay Rate (%)",
    title="Delay Rate Across Markets"
)

fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

fig.update_layout(
    yaxis_title="Delay Rate (%)",
    xaxis_title="Market"
)

st.plotly_chart(fig, use_container_width=True)
highest_market = market_delay.loc[
    market_delay["Delay Rate (%)"].idxmax()
]

st.info(
    f"""
**📌 Business Insight**

The **{highest_market['Market']}** market has the highest shipment delay rate at **{highest_market['Delay Rate (%)']:.1f}%**.

This suggests that logistics planning, carrier performance, or inventory availability should be reviewed for this market.
"""
)
# ---------------------------------------
# Two Charts Side by Side
# ---------------------------------------

left, right = st.columns(2)

# ---------------------------------------
# Customer Segment
# ---------------------------------------

with left:

    st.subheader("👥 Delay Rate by Customer Segment")

    segment_delay = (
        filtered_df
        .groupby("Customer Segment")["Late_delivery_risk"]
        .mean()
        .reset_index()
    )

    segment_delay["Delay Rate (%)"] = (
        segment_delay["Late_delivery_risk"] * 100
    )

    fig_segment = px.bar(
        segment_delay,
        x="Customer Segment",
        y="Delay Rate (%)",
        text="Delay Rate (%)",
        color="Customer Segment"
    )

    fig_segment.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_segment,
        use_container_width=True
    )
# ---------------------------------------
# Delay Pie Chart
# ---------------------------------------

with right:

    st.subheader("🥧 Delay vs On-Time Shipments")

    pie_data = filtered_df["Late_delivery_risk"].value_counts()

    fig_pie = px.pie(
        names=["On-Time", "Delayed"],
        values=[
            pie_data.get(0, 0),
            pie_data.get(1, 0)
        ]
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )
# ==========================================
# Second Row
# ==========================================

left2, right2 = st.columns(2)
with left2:

    st.subheader("📦 Delay Rate by Product Category")

    category_delay = (
        filtered_df
        .groupby("Category Name")["Late_delivery_risk"]
        .mean()
        .reset_index()
    )

    category_delay["Delay Rate (%)"] = (
        category_delay["Late_delivery_risk"] * 100
    )

    fig_category = px.bar(
        category_delay,
        x="Category Name",
        y="Delay Rate (%)",
        color="Category Name",
        text="Delay Rate (%)"
    )

    fig_category.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )
with right2:

    st.subheader("🏢 Delay Rate by Department")

    dept_delay = (
        filtered_df
        .groupby("Department Name")["Late_delivery_risk"]
        .mean()
        .reset_index()
    )

    dept_delay["Delay Rate (%)"] = (
        dept_delay["Late_delivery_risk"] * 100
    )

    fig_dept = px.bar(
        dept_delay,
        x="Department Name",
        y="Delay Rate (%)",
        color="Department Name",
        text="Delay Rate (%)"
    )

    fig_dept.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_dept,
        use_container_width=True
    )