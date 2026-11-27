"""Prefect pipeline automating the whole ML workflow.

Usage:
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow pull | install | quality | test | prepare |
                                      train | evaluate | predict | api | clean
"""

import argparse
import os
import shutil
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    predict_model,
    save_model,
    train_model,
)

DATA_PATH = "breast_cancer.csv"
MODEL_PATH = "model.joblib"
TARGET = "diagnosis"
SOURCES = [
    "model_pipeline.py",
    "main.py",
    "app.py",
    "pipeline_prefect.py",
    "deploiement_prefect.py",
    "test_pipeline.py",
]


def run(cmd):
    """Run a command with the current Python environment; raise on failure."""
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ----------------------------- Tasks ---------------------------------------
@task(name="git-pull")
def git_pull():
    """Retrieve the latest code from the Git repository."""
    run(["git", "pull"])


@task(name="install-dependencies")
def install_dependencies():
    """Install the packages listed in requirements.txt."""
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])


@task(name="code-formatting")
def code_formatting():
    """Format the code with black."""
    run([sys.executable, "-m", "black", "-q", "-l", "100"] + SOURCES)


@task(name="code-quality")
def code_quality():
    """Check style and quality with flake8."""
    run([sys.executable, "-m", "flake8", "--max-line-length", "100"] + SOURCES)


@task(name="code-security")
def code_security():
    """Scan the code for security issues with bandit."""
    run([sys.executable, "-m", "bandit", "-q", "-s", "B104,B101", "-r"] + SOURCES[:3])


@task(name="unit-tests")
def unit_tests():
    """Run the unit tests with pytest."""
    run([sys.executable, "-m", "pytest", "-q", "test_pipeline.py"])


@task(name="data-preparation")
def data_preparation():
    """Load and preprocess the data."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH, TARGET)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


@task(name="model-training")
def model_training(X_train, y_train, scaler):
    """Train the model and save it with its scaler."""
    model = train_model(X_train, y_train)
    save_model(model, MODEL_PATH, scaler)
    print(f"Model saved to {MODEL_PATH}")
    return model


@task(name="model-evaluation")
def model_evaluation(model, X_test, y_test):
    """Evaluate the model on the test set."""
    metrics = evaluate_model(model, X_test, y_test)
    print({k: round(v, 4) for k, v in metrics.items() if k != "confusion_matrix"})
    return metrics


@task(name="prediction")
def prediction(n_samples=5):
    """Predict on a few raw samples using the saved model."""
    import pandas as pd

    model, scaler = load_model(MODEL_PATH)
    df = pd.read_csv(DATA_PATH).sample(n_samples, random_state=0)
    labels, probas = predict_model(model, scaler, df.drop(columns=[TARGET]))
    for true, pred, proba in zip(df[TARGET], labels, probas):
        print(f"true={true} predicted={pred} p(malignant)={proba:.3f}")
    return labels


@task(name="launch-api")
def launch_api(host="0.0.0.0", port=8000):  # nosec B104
    """Start the FastAPI server with uvicorn (blocks until stopped)."""
    run([sys.executable, "-m", "uvicorn", "app:app", "--host", host, "--port", str(port)])


@task(name="clean")
def clean_files():
    """Remove generated files (model and caches)."""
    for path in (MODEL_PATH, "__pycache__", ".pytest_cache"):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        print("removed", path)


# ----------------------------- Flows ---------------------------------------
@flow(name="pull")
def flow_pull():
    """Flow: retrieve the latest code from Git."""
    git_pull()


@flow(name="install")
def flow_install():
    """Flow: install dependencies."""
    install_dependencies()


@flow(name="quality")
def flow_quality():
    """Flow: formatting, quality, security."""
    code_formatting()
    code_quality()
    code_security()


@flow(name="test")
def flow_test():
    """Flow: unit tests."""
    unit_tests()


@flow(name="prepare")
def flow_prepare():
    """Flow: data preparation."""
    return data_preparation()


@flow(name="train")
def flow_train():
    """Flow: prepare data, train and save the model."""
    X_train, _, y_train, _, scaler = data_preparation()
    return model_training(X_train, y_train, scaler)


@flow(name="evaluate")
def flow_evaluate():
    """Flow: evaluate the saved model."""
    _, X_test, _, y_test, _ = data_preparation()
    model, _ = load_model(MODEL_PATH)
    return model_evaluation(model, X_test, y_test)


@flow(name="predict")
def flow_predict():
    """Flow: predict on sample rows."""
    return prediction()


@flow(name="api")
def flow_api():
    """Flow: launch the API server."""
    launch_api()


@flow(name="clean")
def flow_clean():
    """Flow: remove generated files."""
    clean_files()


@flow(name="all")
def flow_all():
    """Flow: the complete pipeline, in order."""
    git_pull()
    install_dependencies()
    code_formatting()
    code_quality()
    code_security()
    unit_tests()
    X_train, X_test, y_train, y_test, scaler = data_preparation()
    model = model_training(X_train, y_train, scaler)
    model_evaluation(model, X_test, y_test)
    prediction()


FLOWS = {
    "all": flow_all,
    "pull": flow_pull,
    "install": flow_install,
    "quality": flow_quality,
    "test": flow_test,
    "prepare": flow_prepare,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
    "api": flow_api,
    "clean": flow_clean,
}


def main():
    """Run the flow selected with --flow."""
    parser = argparse.ArgumentParser(description="Prefect ML pipeline")
    parser.add_argument("--flow", choices=FLOWS, default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
