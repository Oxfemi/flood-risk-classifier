import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import FEATURE_COLUMNS

MODEL_PATH = "models/artifacts/flood_model.pkl"


def load_artifacts():
    """
    Load the trained model and test data.
    """

    model = joblib.load(MODEL_PATH)

    X_test = joblib.load("data/processed/X_test.pkl")
    y_test = joblib.load("data/processed/y_test.pkl")

    return model, X_test, y_test


def get_feature_importance():
    """
    Compute permutation feature importance.
    """

    model, X_test, y_test = load_artifacts()

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
    )

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": result.importances_mean,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    )

    return importance_df








if __name__ == "__main__":

    df = get_feature_importance()

    print(df.head())