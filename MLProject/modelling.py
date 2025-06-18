import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.save_model(model, "model")
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_clean.csv")
    args = parser.parse_args()

    train_model(args.data_path)