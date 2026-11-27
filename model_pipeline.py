"""Modular ML pipeline: data preparation, training, evaluation, persistence.

Dataset: Breast Cancer Wisconsin (Diagnostic), exported to ``breast_cancer.csv``.
Target column: ``diagnosis`` (1 = malignant, 0 = benign).
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(csv_path, target="diagnosis", test_size=0.2, random_state=42):
    """Load the CSV and preprocess it.

    Steps: load, drop duplicates, impute missing values (median), stratified
    train/test split, standardize features (scaler fitted on train only).

    Args:
        csv_path (str): Path to the CSV file.
        target (str): Name of the target column.
        test_size (float): Fraction of rows used for the test set.
        random_state (int): Seed, so the split is reproducible.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")

    df = df.drop_duplicates()
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, n_estimators=200, max_depth=None, random_state=42):
    """Train a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_estimators (int): Number of trees.
        max_depth (int | None): Maximum tree depth.
        random_state (int): Seed for reproducibility.

    Returns:
        RandomForestClassifier: The fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set.

    Args:
        model: Fitted classifier (must implement predict / predict_proba).
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.

    Returns:
        dict: accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))
    return metrics


def save_model(model, path="model.joblib", scaler=None):
    """Save the model (and optionally its scaler) with joblib.

    Args:
        model: Fitted model.
        path (str): Output file path.
        scaler: Fitted scaler to store with the model (optional).

    Returns:
        str: The path the model was saved to.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    return path


def load_model(path="model.joblib"):
    """Load a model saved with save_model().

    Args:
        path (str): Path of the saved file.

    Returns:
        tuple: (model, scaler); scaler is None if none was saved.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("scaler")


def predict_model(model, scaler, features):
    """Predict malignant (1) / benign (0) for new samples.

    Args:
        model: Fitted classifier.
        scaler: Fitted scaler used during training (or None).
        features (pd.DataFrame): Raw (unscaled) feature rows.

    Returns:
        tuple: (labels, probabilities of the malignant class) as lists.
    """
    data = features
    if scaler is not None:
        data = pd.DataFrame(
            scaler.transform(features), columns=features.columns, index=features.index
        )
    labels = model.predict(data).tolist()
    probas = model.predict_proba(data)[:, 1].tolist()
    return labels, probas
