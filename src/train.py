import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

def load_data():
    """
    Load the processed training and testing datasets.
    """

    X_train = joblib.load("data/processed/X_train.pkl")
    X_test = joblib.load("data/processed/X_test.pkl")

    y_train = joblib.load("data/processed/y_train.pkl")
    y_test = joblib.load("data/processed/y_test.pkl")

    return X_train, X_test, y_train, y_test

def get_models():
    """
    Return all models to evaluate.
    """

    return {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),

    }

def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    return {

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    save_classification_report(
        model.__class__.__name__,
        y_test,
        predictions,
    )

    save_confusion_matrix(
        model.__class__.__name__,
        y_test,
        predictions,
    )

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    return {

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

def compare_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate all models.
    """

    results = []

    best_model = None
    best_score = 0

    for name, model in models.items():

        print(f"\nTraining {name}...")

        trained_model = train_model(
            model,
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            trained_model,
            X_test,
            y_test,
        )

        metrics["Model"] = name

        results.append(metrics)

        if metrics["Accuracy"] > best_score:

            best_score = metrics["Accuracy"]
            best_model = trained_model

    results_df = pd.DataFrame(results)

    return results_df, best_model

def save_results(results_df, best_model):
    """
    Save model comparison results
    and the best trained model.
    """

    Path("results").mkdir(exist_ok=True)
    Path("models/artifacts").mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        "results/model_results.csv",
        index=False,
    )

    joblib.dump(
        best_model,
        "models/artifacts/flood_model.pkl",
    )

def create_result_directories():

    Path("results").mkdir(exist_ok=True)

    Path(
        "results/classification_reports"
    ).mkdir(parents=True, exist_ok=True)

    Path(
        "results/confusion_matrices"
    ).mkdir(parents=True, exist_ok=True)

def save_classification_report(
    model_name,
    y_test,
    predictions,
):
    """
    Save the classification report
    as a text file.
    """

    report = classification_report(
        y_test,
        predictions,
    )

    filename = (
        f"results/classification_reports/"
        f"{model_name}.txt"
    )

    with open(filename, "w") as file:
        file.write(report)

def save_confusion_matrix(
    model_name,
    y_test,
    predictions,
):
    """
    Save confusion matrix as an image.
    """

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        f"results/confusion_matrices/"
        f"{model_name}.png"
    )

    plt.savefig(filename)

    plt.close()





def main():

    print("=" * 50)
    print("Flood Risk Classifier Training")
    print("=" * 50)

    create_result_directories()

    X_train, X_test, y_train, y_test = load_data()

    models = get_models()

    results_df, best_model = compare_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_results(
        results_df,
        best_model,
    )

    print("\nTraining Complete!\n")

    print(results_df.sort_values(
        by="Accuracy",
        ascending=False,
    ))

if __name__ == "__main__":
    main()