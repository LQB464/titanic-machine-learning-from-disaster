# src/main.py

from __future__ import annotations

import argparse
from pathlib import Path

from .config import OUTPUT_DIR
from .data import load_raw_data
from .training import (
    run_baseline_experiments,
    tune_best_knn,
    train_final_model_and_predict_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Titanic pipeline runner. "
            "Runs baseline models, tunes best model, and creates Kaggle submission."
        )
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline experiments and go directly to tuning.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Run only baseline experiments and skip hyperparameter tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save results and submission file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    df_train = load_raw_data()

    # 1. Baseline experiments
    if not args.skip_baseline:
        print("Running baseline experiments...")
        baseline_df = run_baseline_experiments(df_train)
        baseline_path = output_dir / "baseline_results.csv"
        baseline_df.to_csv(baseline_path, index=False)
        print(f"Baseline results saved to {baseline_path}")
        print(baseline_df)
    else:
        print("Skipping baseline experiments as requested.")

    # Nếu chỉ muốn chạy baseline rồi dừng
    if args.skip_tuning:
        print("Skipping tuning and final submission generation as requested.")
        return

    # 2. Tune best model (KNN theo thiết kế hiện tại)
    print("Tuning best model (KNN)...")
    grid_knn = tune_best_knn(df_train)

    # 3. Train final model on full train data and predict test
    print("Training final model on full training data and predicting test set...")
    submission = train_final_model_and_predict_test(grid_knn)

    submission_path = output_dir / "submission_knn.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")


if __name__ == "__main__":
    main()