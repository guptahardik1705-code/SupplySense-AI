import streamlit as st

st.set_page_config(
    page_title="SupplySense AI",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 SupplySense AI")

st.subheader("AI-Powered Shipment Delay Prediction System")

st.markdown("---")

st.write("""
Welcome to **SupplySense AI**, an end-to-end Machine Learning application that predicts shipment delays before dispatch using Machine Learning and Explainable AI.

### 🚀 Features

- 📊 Executive Dashboard
- 🤖 Shipment Delay Prediction
- 🔍 SHAP Explainability
- 🚨 High Risk Shipment Monitoring

### 🧠 Machine Learning Model

- XGBoost Classifier
- Feature Engineering Pipeline
- Explainable AI using SHAP

### 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- XGBoost
- SHAP
- Pandas
- NumPy

---

### 📌 How to Use

Use the **left sidebar** to navigate through the application.

1. **Dashboard** – View shipment statistics and KPIs.
2. **Prediction** – Predict shipment delay for a single shipment.
3. **SHAP Analysis** – Understand why the model made its prediction.
4. **High Risk Shipments** – Review shipments with the highest predicted risk.
5. **About** – Learn more about the project.
""")

st.success("✅ SupplySense AI is ready to use.")