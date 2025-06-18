import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Argumen dari MLProject
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="diabetes_clean.csv")
args = parser.parse_args()

# MLflow autolog
mlflow.sklearn.autolog()

# Tracking setup (opsional jika pakai default)
mlflow.set_experiment("diabetes_prediction_experiment")

# Load data
df = pd.read_csv(args.data_path)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

with mlflow.start_run(nested=True):
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)
    print(f"Accuracy: {acc:.4f}")