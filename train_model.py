import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "student_data.csv"
MODEL_PATH = BASE_DIR / "model" / "student_model.pkl"
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_mark",
    "study_hours",
    "completed_assignments",
]


def train_model():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    if data.empty:
        raise ValueError("Dataset is empty. Add student records before training.")

    X = data[FEATURES]
    y = data["performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    print(f"Model trained successfully and saved to {MODEL_PATH}")
    print(f"Model accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    train_model()
