"""Entry point: run the pipeline steps from the command line.

Examples:
    python main.py prepare
    python main.py train
    python main.py evaluate
    python main.py all
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Modular ML pipeline")
    parser.add_argument(
        "step",
        choices=["prepare", "train", "evaluate", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument("--data", default="breast_cancer.csv", help="CSV data file")
    parser.add_argument("--target", default="diagnosis", help="Target column")
    parser.add_argument("--model-path", default="model.joblib", help="Model file")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, X_test, y_train, y_test, scaler = prepare_data(
        args.data, args.target, args.test_size, args.seed
    )

    if args.step == "prepare":
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Class balance (train):\n{y_train.value_counts(normalize=True)}")
        return

    if args.step in ("train", "all"):
        model = train_model(X_train, y_train, args.n_estimators, args.max_depth, args.seed)
        path = save_model(model, args.model_path, scaler)
        print(f"Model trained and saved to {path}")

    if args.step in ("evaluate", "all"):
        model, _ = load_model(args.model_path)
        metrics = evaluate_model(model, X_test, y_test)
        print("Metrics:", {k: round(v, 4) for k, v in metrics.items() if k != "confusion_matrix"})
        print("Confusion matrix:", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
