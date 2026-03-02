"""
Standalone training script.
Purpose: Load CSV, train model, save to model/model.pkl.
Modify: Adapt LOAD DATA, PREPROCESS, TRAIN sections to your dataset.

TODO: Update file path, target column, and feature columns for your data.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Path to CSV. Default: ../../data/your_data.csv
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent.parent / "data"
CSV_FILE = DATA_PATH / "sample_data.csv"  # TODO: Replace with your CSV filename

MODEL_DIR = SCRIPT_DIR.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

# TODO: Set these for your dataset
TARGET_COLUMN = "target"  # Column to predict
FEATURE_COLUMNS = None  # None = use all except target; or list of column names


def main():
    # -------- LOAD DATA --------
    if not CSV_FILE.exists():
        print(f"CSV not found at {CSV_FILE}")
        print("Place your CSV in the data/ folder and update CSV_FILE in train.py")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # -------- PREPROCESS --------
    # TODO: Handle missing values, encoding, scaling
    df = df.dropna()

    if TARGET_COLUMN not in df.columns:
        print(f"Target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")
        sys.exit(1)

    X = df.drop(columns=[TARGET_COLUMN]) if FEATURE_COLUMNS is None else df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Encode if needed (sklearn classifiers expect numeric targets)
    if y.dtype == "object" or y.dtype.name == "category":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Save encoder for inference if needed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------- TRAIN --------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # -------- EVALUATE --------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(X.columns, model.feature_importances_))
        print("Feature importances:", fi)

    # -------- SAVE --------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
