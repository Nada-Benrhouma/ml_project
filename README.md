# ML Pipeline – Workshop 2: Code Modularization

Dataset: Breast Cancer Wisconsin (Diagnostic) – 569 rows, 30 numeric features, target `diagnosis` (1 = malignant).

## Setup
    pip install -r requirements.txt

## Usage
    python main.py prepare     # load + preprocess, print shapes
    python main.py train       # prepare, train, save model.joblib
    python main.py evaluate    # load model.joblib, print metrics
    python main.py all         # train then evaluate

Options: `--data`, `--target`, `--model-path`, `--n-estimators`, `--max-depth`, `--test-size`, `--seed`.

## Files
- `model_pipeline.py`: prepare_data, train_model, evaluate_model, save_model, load_model
- `main.py`: CLI entry point
