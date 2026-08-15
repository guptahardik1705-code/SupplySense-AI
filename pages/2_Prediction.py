import streamlit as st
import pandas as pd
import joblib

from utils.feature_engineering import prepare_features


st.set_page_config(
    page_title="Shipment Prediction",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------
# Load Model and Data
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/shipment_delay_pipeline.pkl"
    )


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/shipment_clean_data.csv"
    )


model = load_model()
df = load_data()


# -------------------------------
# Title
# -------------------------------

st.title("🤖 Shipment Delay Prediction")

st.write(
    "Predict shipment delay risk using Machine Learning"
)

st.markdown("---")


# -------------------------------
# Input Section
# -------------------------------

st.subheader("Enter Shipment Details")


col1, col2 = st.columns(2)


with col1:

    shipment_type = st.selectbox(
        "Shipment Type",
        sorted(df["Type"].unique())
    )

    market = st.selectbox(
        "Market",
        sorted(df["Market"].unique())
    )

    customer_segment = st.selectbox(
        "Customer Segment",
        sorted(df["Customer Segment"].unique())
    )

    department = st.selectbox(
        "Department Name",
        sorted(df["Department Name"].unique())
    )

    category = st.selectbox(
        "Category Name",
        sorted(df["Category Name"].unique())
    )

    shipping_mode = st.selectbox(
        "Shipping Mode",
        sorted(df["Shipping Mode"].unique())
    )

    order_region = st.selectbox(
        "Order Region",
        sorted(df["Order Region"].unique())
    )


with col2:

    scheduled_days = st.number_input(
        "Scheduled Shipping Days",
        min_value=1,
        max_value=30,
        value=4
    )

    quantity = st.number_input(
        "Order Quantity",
        min_value=1,
        value=1
    )

    product_price = st.number_input(
        "Product Price",
        value=50.0
    )

    sales = st.number_input(
        "Sales",
        value=100.0
    )

    discount = st.number_input(
        "Order Discount",
        value=0.0
    )

    discount_rate = st.number_input(
        "Discount Rate",
        value=0.0
    )

    profit = st.number_input(
        "Profit Per Order",
        value=20.0
    )

    benefit = st.number_input(
        "Benefit Per Order",
        value=20.0
    )


st.markdown("---")


# -------------------------------
# Prediction
# -------------------------------

if st.button("🚀 Predict Shipment Risk"):


    input_data = prepare_features(
        shipment_type,
        scheduled_days,
        benefit,
        sales,
        category,
        customer_segment,
        department,
        market,
        discount,
        discount_rate,
        product_price,
        quantity,
        profit,
        order_region,
        shipping_mode
    )


    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]


    # Save prediction for SHAP page

    st.session_state["shipment_features"] = input_data
    st.session_state["shipment_probability"] = probability
    st.session_state["shipment_prediction"] = prediction[0]


    st.markdown("---")

    st.subheader("Prediction Result")


    if prediction[0] == 1:

        st.error(
            "⚠ Shipment is likely to be DELAYED"
        )

    else:

        st.success(
            "✅ Shipment is likely to be ON TIME"
        )


    # Risk Level

    if probability < 0.30:

        risk = "🟢 Low Risk"


    elif probability < 0.60:

        risk = "🟡 Medium Risk"


    else:

        risk = "🔴 High Risk"


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Delay Probability",
            f"{probability*100:.2f}%"
        )


    with col2:

        st.metric(
            "Operational Risk",
            risk
        )


    st.markdown("---")


    if risk == "🔴 High Risk":

        st.error(
            """
            **Recommended Actions**

            - Review logistics partner
            - Consider faster shipping mode
            - Monitor shipment closely
            - Inform customer proactively
            """
        )


    elif risk == "🟡 Medium Risk":

        st.warning(
            """
            **Recommended Actions**

            - Track shipment progress
            - Monitor carrier performance
            - Keep backup delivery options ready
            """
        )


    else:

        st.info(
            """
            **Recommendation**

            Shipment appears low risk.
            Continue with normal delivery process.
            """
        )


    st.markdown("---")


    st.page_link(
        "pages/4_SHAP_Analysis.py",
        label="🔍 Explain This Prediction",
        icon="🔍"
    )