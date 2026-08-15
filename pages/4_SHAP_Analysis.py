import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="SHAP Analysis",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------
# Load Pipeline
# --------------------------------

@st.cache_resource
def load_pipeline():

    return joblib.load(
        "models/shipment_delay_pipeline.pkl"
    )


pipeline = load_pipeline()



# --------------------------------
# Title
# --------------------------------

st.title("🔍 SHAP Explainability Analysis")

st.write(
    """
    SHAP explains how each feature influenced the
    shipment delay prediction.
    """
)


st.markdown("---")



# --------------------------------
# Check Prediction
# --------------------------------

if "shipment_features" not in st.session_state:

    st.warning(
        "Please make a prediction first from the Prediction page."
    )

    st.stop()



input_data = st.session_state["shipment_features"]

probability = st.session_state["shipment_probability"]

prediction = st.session_state["shipment_prediction"]




# --------------------------------
# Prediction Summary
# --------------------------------

st.subheader("📦 Prediction Summary")


col1, col2 = st.columns(2)


with col1:

    if prediction == 1:

        st.error("Shipment Likely Delayed")

    else:

        st.success("Shipment Likely On Time")



with col2:

    st.metric(
        "Delay Probability",
        f"{probability*100:.2f}%"
    )



st.markdown("---")



# --------------------------------
# SHAP Calculation
# --------------------------------

st.subheader("📊 Feature Contribution")


try:


    preprocessor = pipeline.named_steps["preprocessor"]

    classifier = pipeline.named_steps["classifier"]



    # Transform input

    transformed_input = preprocessor.transform(
        input_data
    )


    # Feature names

    feature_names = (
        preprocessor.get_feature_names_out()
    )


    # Clean names

    clean_feature_names = (
        pd.Series(feature_names)
        .str.replace(
            "cat__",
            "",
            regex=False
        )
        .str.replace(
            "remainder__",
            "",
            regex=False
        )
        .str.replace(
            "Days for shipment (scheduled)",
            "Scheduled Shipping Days",
            regex=False
        )
        .str.replace(
            "Type",
            "Shipment Type",
            regex=False
        )
        .tolist()
    )



    # SHAP

    explainer = shap.TreeExplainer(
        classifier
    )


    shap_values = explainer(
        transformed_input
    )



    # Create clean explanation

    waterfall_explanation = shap.Explanation(

        values=shap_values.values[0],

        base_values=shap_values.base_values[0],

        data=transformed_input[0],

        feature_names=clean_feature_names

    )



    fig = plt.figure(
        figsize=(10,6)
    )


    shap.plots.waterfall(
        waterfall_explanation,
        show=False
    )


    st.pyplot(
        fig,
        clear_figure=True
    )



except Exception as e:

    st.error(
        f"SHAP Error: {e}"
    )

    st.stop()



# --------------------------------
# Business Interpretation
# --------------------------------


st.markdown("---")

st.subheader("📌 Business Interpretation")



explanation = pd.DataFrame(

    {

        "Feature": clean_feature_names,

        "SHAP Value": shap_values.values[0]

    }

)



explanation["Impact Direction"] = explanation[
    "SHAP Value"
].apply(

    lambda x:

    "🔴 Increasing Delay Risk"

    if x > 0

    else

    "🟢 Reducing Delay Risk"

)



explanation["Impact Strength"] = explanation[
    "SHAP Value"
].abs().apply(

    lambda x:

    "High"

    if x > 0.5

    else

    "Medium"

    if x > 0.2

    else

    "Low"

)



explanation = explanation.sort_values(

    "SHAP Value",

    key=abs,

    ascending=False

)



st.dataframe(

    explanation.head(10),

    use_container_width=True

)



# --------------------------------
# Main Drivers
# --------------------------------


st.markdown("---")

st.subheader("🚨 Main Delay Drivers")



risk_features = explanation[

    explanation["SHAP Value"] > 0

].head(3)



if len(risk_features) == 0:

    st.info(
        "No major delay increasing factors found."
    )


else:

    for _, row in risk_features.iterrows():

        st.error(

            f"""
**{row['Feature']}**

Impact Strength: {row['Impact Strength']}

Effect: Increasing Delay Risk

"""

        )



st.subheader("🟢 Factors Reducing Delay Risk")



safe_features = explanation[

    explanation["SHAP Value"] < 0

].head(3)



if len(safe_features) == 0:

    st.info(
        "No major risk reducing factors found."
    )


else:

    for _, row in safe_features.iterrows():

        st.success(

            f"""
**{row['Feature']}**

Impact Strength: {row['Impact Strength']}

Effect: Reducing Delay Risk

"""

        )