import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="High Risk Shipments",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 High Risk Shipments")

st.markdown("---")

@st.cache_resource
def load_objects():
    with open("models/project_objects.pkl", "rb") as file:
        return pickle.load(file)

objects = load_objects()

xgb_prob = objects["xgb_prob"]
xgb_pred = objects["xgb_pred"]

# Create DataFrame
risk_df = pd.DataFrame({
    "Shipment": range(1, len(xgb_prob) + 1),
    "Delay Probability": xgb_prob
})

# Convert probability to %
risk_df["Delay Probability (%)"] = (
    risk_df["Delay Probability"] * 100
).round(2)

# Risk Level
risk_df["Risk Level"] = risk_df["Delay Probability"].apply(
    lambda x:
        "🔴 High" if x >= 0.80 else
        "🟠 Medium" if x >= 0.50 else
        "🟢 Low"
)

risk_df = risk_df.sort_values(
    by="Delay Probability",
    ascending=False
)

st.subheader("Top High Risk Shipments")

st.dataframe(
    risk_df[
        ["Shipment",
         "Delay Probability (%)",
         "Risk Level"]
    ],
    use_container_width=True
)