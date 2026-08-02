import joblib
import numpy as np

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

def prepare_input(features, scaler):
    """
    Prepare input for prediction.
    """

    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    return features

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

    sample = [
        3,
        8,
        6,
        6,
        4,
        4,
        6,
        2,
        3,
        2,
        5,
        10,
        7,
        4,
        2,
        3,
        4,
        3,
        2,
        6,
    ]

    result = predict(sample)

    print(result)

