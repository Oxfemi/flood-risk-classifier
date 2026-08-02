import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# ==========================
# Configuration
# ==========================

DATA_PATH = "data/raw/flood.csv"

LOW_THRESHOLD = 0.475
HIGH_THRESHOLD = 0.520

def load_data():
    """
    Load the raw flood dataset.
    """
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df

def load_data():
    """
    Load the raw flood dataset.
    """
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df

def classify_risk(probability):
    """
    Convert FloodProbability into
    Low, Medium and High risk classes.
    """

    if probability < LOW_THRESHOLD:
        return "Low"

    elif probability < HIGH_THRESHOLD:
        return "Medium"

    else:
        return "High"

def create_target(df):
    """
    Create the FloodRisk column.
    """

    df["FloodRisk"] = df["FloodProbability"].apply(classify_risk)

    return df

def split_features_target(df):
    """
    Split the dataset into features (X)
    and target (y).
    """

    X = df.drop(columns=["FloodProbability", "FloodRisk"])
    y = df["FloodRisk"]

    return X, y

def encode_target(y):
    """
    Encode Low, Medium and High
    into numerical values.
    """

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    return y_encoded, encoder

def split_data(X, y):
    """
    Split the dataset into
    training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale the numerical features.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def save_artifacts(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
):
    """
    Save processed data and preprocessing artifacts.
    """

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train, "data/processed/X_train.pkl")
    joblib.dump(X_test, "data/processed/X_test.pkl")

    joblib.dump(y_train, "data/processed/y_train.pkl")
    joblib.dump(y_test, "data/processed/y_test.pkl")

    joblib.dump(scaler, "models/artifacts/scaler.pkl")
    joblib.dump(encoder, "models/artifacts/label_encoder.pkl")




def main():

    df = load_data()

    df = create_target(df)

    df.to_csv(
    "data/processed/flood_clean.csv",
    index=False,
)

    X, y = split_features_target(df)

    y, encoder = encode_target(y)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test,
    )

    save_artifacts(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder,
    )

    print("\nPreprocessing completed successfully!")

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")


if __name__ == "__main__":
    main()

    