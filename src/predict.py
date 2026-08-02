import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/artifacts/flood_model.pkl"
SCALER_PATH = "models/artifacts/scaler.pkl"
ENCODER_PATH = "models/artifacts/label_encoder.pkl"

def load_artifacts():
    """
    Load the trained model and preprocessing objects.
    """

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder


FEATURE_COLUMNS = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]


def prepare_input(features, scaler):
    """
    Convert a dictionary of feature values
    into a scaled DataFrame.
    """

    input_df = pd.DataFrame([features])

    input_df = input_df[FEATURE_COLUMNS]

    scaled = scaler.transform(input_df)

    return scaled

def predict(features):

    model, scaler, encoder = load_artifacts()

    processed = prepare_input(features, scaler)

    prediction = model.predict(processed)[0]

    probabilities = model.predict_proba(processed)[0]

    label = encoder.inverse_transform([prediction])[0]

    confidence = float(np.max(probabilities))

    probability_map = {
        label_name: float(prob)
        for label_name, prob in zip(
            encoder.classes_,
            probabilities,
        )
    }

    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": probability_map,
    }












if __name__ == "__main__":

    sample = {
        "MonsoonIntensity": 3,
        "TopographyDrainage": 8,
        "RiverManagement": 6,
        "Deforestation": 6,
        "Urbanization": 4,
        "ClimateChange": 4,
        "DamsQuality": 6,
        "Siltation": 2,
        "AgriculturalPractices": 3,
        "Encroachments": 2,
        "IneffectiveDisasterPreparedness": 5,
        "DrainageSystems": 10,
        "CoastalVulnerability": 7,
        "Landslides": 4,
        "Watersheds": 2,
        "DeterioratingInfrastructure": 3,
        "PopulationScore": 4,
        "WetlandLoss": 3,
        "InadequatePlanning": 2,
        "PoliticalFactors": 6,
    }

  
    result = predict(sample)

    print(result)

