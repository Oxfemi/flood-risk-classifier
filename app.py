import pandas as pd
import streamlit as st
import re

from src.predict import predict
from src.config import FEATURE_COLUMNS
from src.explain import get_feature_importance


st.set_page_config(
    page_title="Flood Risk Classifier",
    page_icon="🌊",
    layout="wide",
)

col1, col2 = st.columns([1, 6])

with col1:
    st.image(
        "https://img.icons8.com/color/96/floods.png",
        width=70,
    )

with col2:
    st.title("Flood Risk Classifier")
    st.caption("AI/ML Capstone Project | 3MTT NextGen Cohort")



with st.container():

    st.info(
        """
🌊 **Flood Risk Classifier**

This AI application predicts flood risk using environmental,
climatic and infrastructure indicators.

Move the sliders in the sidebar to simulate different
conditions and click **Predict Flood Risk**.
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


st.sidebar.title("🌊 Flood Inputs")

st.sidebar.markdown(
    """
Adjust the environmental
conditions below.
"""
)

st.sidebar.divider()


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


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Features",
        len(FEATURE_COLUMNS),
    )

with col2:
    st.metric(
        "ML Model",
        "Random Forest",
    )

with col3:
    st.metric(
        "Prediction Classes",
        "3",
    )


with st.expander("📊 Current Input Summary"):

    input_df = pd.DataFrame(
        user_input.items(),
        columns=["Feature", "Value"],
    )

    st.dataframe(
        input_df,
        use_container_width=True,
    )



if st.button("🌊 Predict Flood Risk", use_container_width=True):

    result = predict(user_input)

    risk = result["prediction"]

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Prediction Result")
        st.divider()
        if risk == "Low":
            st.success("🟢 Low Flood Risk")
        elif risk == "Medium":
            st.warning("🟡 Medium Flood Risk")
        else:
            st.error("🔴 High Flood Risk")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                "Risk Level",
                risk,
            )

        with colB:
            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}",
            )

        st.progress(result["confidence"])

        st.write(
            f"Model confidence: **{result['confidence']:.2%}**"
        )

        st.subheader("Recommendation")

        if risk == "Low":
            st.success(
                """
            ### Recommended Actions
            - Continue monitoring rainfall.
            - Maintain drainage systems.
            - Preserve surrounding vegetation.
            """
            )
        elif risk == "Medium":
            st.warning(
                """
            ### Recommended Actions
            - Inspect drainage channels.
            - Monitor weather forecasts.
            - Prepare emergency supplies.
            """
            )
        else:
            st.error(
                """
            ### Immediate Actions

            - Activate emergency response plans.
            - Warn nearby communities.
            - Prepare for possible evacuation.
            """
            )
    with right_col:
        st.subheader("Prediction Probabilities")
        st.divider()
    
        probability_df = pd.DataFrame.from_dict(
            result["probabilities"],
            orient="index",
            columns=["Probability"],
        )
    
        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False,
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

        st.divider()

        st.subheader("📊 Model Explainability")

        st.caption(
            "These are the features that had the greatest influence on the trained model."
        )

        importance_df = get_feature_importance()

        st.bar_chart(
            importance_df.set_index("Feature")
        )

        st.subheader("🏆 Top 5 Most Important Features")

        st.dataframe(
            importance_df.head(),
            use_container_width=True,
        )

        st.info(
            """
        **What does Feature Importance mean?**
        
        Feature importance measures how much each variable contributed to the model's predictions.
        
        Higher importance means the model relied more heavily on that feature when determining flood risk.
        """
        )





st.divider()

st.caption(
    "Built with ❤️ using Python, Scikit-learn and Streamlit"
)
