import pandas as pd
import streamlit as st
import re

from src.predict import predict
from src.config import FEATURE_COLUMNS


st.set_page_config(
    page_title="Flood Risk Classifier",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Flood Risk Classifier")
st.caption("AI/ML Capstone Project | 3MTT NextGen Cohort")

with st.expander("📖 About this Project"):
    st.write(
        """
        This application predicts flood risk using a trained
        Machine Learning model.

        Adjust the environmental and infrastructure factors
        using the sliders in the sidebar and click the button
        below to predict flood risk.
        """
    )


FEATURE_ICONS = {
    "MonsoonIntensity": "🌧️",
    "TopographyDrainage": "🏞️",
    "RiverManagement": "🌊",
    "Deforestation": "🌳",
    "Urbanization": "🏙️",
    "ClimateChange": "🌡️",
    "DamsQuality": "🛑",
    "Siltation": "🪨",
    "AgriculturalPractices": "🌾",
    "Encroachments": "🏠",
    "IneffectiveDisasterPreparedness": "🚨",
    "DrainageSystems": "🚰",
    "CoastalVulnerability": "🏖️",
    "Landslides": "⛰️",
    "Watersheds": "💧",
    "DeterioratingInfrastructure": "🏗️",
    "PopulationScore": "👥",
    "WetlandLoss": "🦆",
    "InadequatePlanning": "📋",
    "PoliticalFactors": "🏛️",
}


st.sidebar.header("Input Features")


def get_user_input():

    data = {}

    for feature in FEATURE_COLUMNS:

        icon = FEATURE_ICONS.get(feature, "📌")

        label = f"{icon} {re.sub(r'(?<!^)(?=[A-Z])', ' ', feature)}"

        data[feature] = st.sidebar.slider(
            label,
            min_value=1,
            max_value=10,
            value=5,
        )

    return data


user_input = get_user_input()

if st.button("🌊 Predict Flood Risk", use_container_width=True):

    result = predict(user_input)

    risk = result["prediction"]

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Prediction Result")
        if risk == "Low":
            st.success("🟢 Low Flood Risk")
        elif risk == "Medium":
            st.warning("🟡 Medium Flood Risk")
        else:
            st.error("🔴 High Flood Risk")

        st.metric(
            "Prediction Confidence",
            f"{result['confidence']:.2%}",
        )

        st.subheader("Recommendation")

        if risk == "Low":
            st.info("Flood risk is currently low. Continue routine monitoring.")
        elif risk == "Medium":
            st.warning(
                "Flood risk is moderate. Monitor weather forecasts and inspect drainage systems."
            )
        else:
            st.error(
                "High flood risk detected. Emergency preparedness is advised."
            )
    with right_col:
        st.subheader("Prediction Probabilities")
    
        probability_df = pd.DataFrame.from_dict(
            result["probabilities"],
            orient="index",
            columns=["Probability"],
        )
    
        st.bar_chart(probability_df)
    
        st.subheader("Selected Inputs")
    
        st.dataframe(
            pd.DataFrame(
                user_input.items(),
                columns=["Feature", "Value"],
            ),
            use_container_width=True,
        )


