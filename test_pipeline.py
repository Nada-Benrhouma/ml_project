"""Unit tests for model_pipeline.py (run with: pytest -v)."""

import pytest

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA = "breast_cancer.csv"


@pytest.fixture(scope="module")
def data():
    return prepare_data(DATA)


@pytest.fixture(scope="module")
def model(data):
    X_train, _, y_train, _, _ = data
    return train_model(X_train, y_train, n_estimators=50)


def test_prepare_data_shapes(data):
    X_train, X_test, y_train, y_test, _ = data
    assert len(X_train) + len(X_test) == 569
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_prepare_data_no_missing_values(data):
    X_train, X_test, _, _, _ = data
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()


def test_prepare_data_class_balance_preserved(data):
    _, _, y_train, y_test, _ = data
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_prepare_data_missing_file():
    with pytest.raises(FileNotFoundError):
        prepare_data("does_not_exist.csv")


def test_prepare_data_wrong_target():
    with pytest.raises(ValueError):
        prepare_data(DATA, target="wrong_column")


def test_train_model_is_fitted(model):
    assert model.n_estimators == 50
    assert hasattr(model, "feature_importances_")


def test_evaluate_model_metrics(model, data):
    _, X_test, _, y_test, _ = data
    metrics = evaluate_model(model, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0 <= metrics[key] <= 1
    assert len(metrics["confusion_matrix"]) == 2


def test_model_quality_threshold(model, data):
    _, X_test, _, y_test, _ = data
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] >= 0.90
    assert metrics["recall"] >= 0.85


def test_reproducibility(data):
    X_train, X_test, y_train, _, _ = data
    m1 = train_model(X_train, y_train, n_estimators=20, random_state=1)
    m2 = train_model(X_train, y_train, n_estimators=20, random_state=1)
    assert (m1.predict(X_test) == m2.predict(X_test)).all()


def test_save_load_roundtrip(model, data, tmp_path):
    _, X_test, _, _, scaler = data
    path = save_model(model, str(tmp_path / "m.joblib"), scaler)
    loaded, loaded_scaler = load_model(path)
    assert (model.predict(X_test) == loaded.predict(X_test)).all()
    assert loaded_scaler is not None


def test_load_missing_model():
    with pytest.raises(FileNotFoundError):
        load_model("no_such_model.joblib")


def test_predict_model(model, data):
    from model_pipeline import predict_model

    _, X_test, _, _, _ = data
    labels, probas = predict_model(model, None, X_test.head(3))
    assert len(labels) == 3
    assert all(0 <= p <= 1 for p in probas)


def test_api_predict(tmp_path, monkeypatch, model, data):
    import pandas as pd
    from fastapi.testclient import TestClient

    import app as api

    _, _, _, _, scaler = data
    path = save_model(model, str(tmp_path / "m.joblib"), scaler)
    monkeypatch.setattr(api, "MODEL_PATH", path)
    client = TestClient(api.app)

    row = pd.read_csv(DATA).drop(columns=["diagnosis"]).iloc[0].to_dict()
    response = client.post("/predict", json={"samples": [row]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["diagnosis"] in ("malignant", "benign")
    assert client.post("/predict", json={"samples": [{"x": 1.0}]}).status_code == 422
