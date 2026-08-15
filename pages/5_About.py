import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About SupplySense AI")

st.markdown("---")

st.write("""
**SupplySense AI** is an end-to-end Machine Learning application designed to predict shipment delays before dispatch. It combines predictive analytics with Explainable AI (SHAP) to provide accurate predictions along with clear business-focused explanations.
""")

st.markdown("### 🎯 Project Objective")

st.write("""
The objective of this project is to help logistics and supply chain teams proactively identify shipments that are likely to be delayed, enabling better operational planning and decision-making.
""")

st.markdown("### 🤖 Machine Learning Model")

st.write("""
- **Algorithm:** XGBoost Classifier
- **Problem Type:** Binary Classification
- **Target Variable:** Late Delivery Risk
- **Feature Engineering:** Custom preprocessing and business feature engineering pipeline
- **Explainability:** SHAP (SHapley Additive Explanations)
""")

st.markdown("### 📊 Application Features")

st.write("""
- 📊 Executive Dashboard
- 🤖 Shipment Delay Prediction
- 🔍 SHAP Explainability
- 🚨 High-Risk Shipment Identification
""")

st.markdown("### ⚙️ Technologies Used")

st.write("""
- Python
- Streamlit
- XGBoost
- Scikit-learn
- SHAP
- Pandas
- NumPy
- Matplotlib
""")

st.markdown("### 📁 Dataset")

st.write("""
The application uses a supply chain shipment dataset containing order, customer, product, shipping, and logistics information. The data is processed through a feature engineering pipeline before being used for prediction.
""")

st.markdown("### 💡 Business Value")

st.write("""
SupplySense AI enables organizations to:

- Predict shipment delays before dispatch
- Identify high-risk shipments
- Understand the key factors influencing each prediction
- Support data-driven logistics and supply chain decisions
""")