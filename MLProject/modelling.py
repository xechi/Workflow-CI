import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="diabetes_clean.csv")
args = parser.parse_args()

# Autolog
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv(args.data_path)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate and log
acc = model.score(X_test, y_test)
mlflow.log_metric("accuracy", acc)

# Save model
mlflow.sklearn.log_model(model, "model")

print(f"Accuracy: {acc:.4f}")