# src/training.py
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from .config import model_cfg, data_cfg
from .data import load_raw_data, load_test_data, split_train_valid
from .features import full_feature_engineering
from .modeling import (
    build_base_models,
    build_pipeline_for_model,
    build_pca_pipeline,
    build_voting_classifier,
    get_knn_param_grid,
)


def evaluate_model(pipe, X_train, y_train) -> Tuple[float, float]:
    """Tính mean accuracy và f1 macro bằng cross val."""
    acc = cross_val_score(pipe, X_train, y_train, cv=model_cfg.cv_folds, scoring="accuracy")
    f1 = cross_val_score(pipe, X_train, y_train, cv=model_cfg.cv_folds, scoring="f1")
    return acc.mean(), f1.mean()


def run_baseline_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Approach 1: các model baseline, có thể thêm phiên bản PCA."""
    X_train, X_valid, y_train, y_valid = split_train_valid(df, target_col=data_cfg.target_col)

    # Feature engineering trước khi đưa vào pipeline
    X_train_fe = full_feature_engineering(X_train)
    X_valid_fe = full_feature_engineering(X_valid)

    base_models = build_base_models()
    rows = []

    for name, model in base_models.items():
        pipe = build_pipeline_for_model(model)
        pipe.fit(X_train_fe, y_train)
        y_pred = pipe.predict(X_valid_fe)

        acc = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)

        rows.append({"model": name, "accuracy_valid": acc, "f1_valid": f1})

    return pd.DataFrame(rows)


def tune_best_knn(df: pd.DataFrame) -> GridSearchCV:
    """Tuning cho KNN với top feature set, giống phần Approach 2."""
    X_train, X_valid, y_train, y_valid = split_train_valid(df, target_col=data_cfg.target_col)

    X_train_fe = full_feature_engineering(X_train)
    pipe_knn = build_pipeline_for_model(build_base_models()["knn"])
    param_grid = get_knn_param_grid()

    grid = GridSearchCV(
        estimator=pipe_knn,
        param_grid=param_grid,
        cv=model_cfg.cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_fe, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    X_valid_fe = full_feature_engineering(X_valid)
    y_pred = grid.predict(X_valid_fe)
    print("Valid accuracy:", accuracy_score(y_valid, y_pred))
    print("Valid f1:", f1_score(y_valid, y_pred))

    return grid


def train_final_model_and_predict_test(grid: GridSearchCV) -> pd.DataFrame:
    """Dùng best estimator huấn luyện lại trên full train, dự đoán test, tạo submission."""
    df_train = load_raw_data()
    df_test = load_test_data()

    X_full = df_train.drop(columns=[data_cfg.target_col])
    y_full = df_train[data_cfg.target_col]

    X_full_fe = full_feature_engineering(X_full)
    X_test_fe = full_feature_engineering(df_test)

    best_pipe = grid.best_estimator_
    best_pipe.fit(X_full_fe, y_full)

    test_pred = best_pipe.predict(X_test_fe)

    submission = pd.DataFrame(
        {
            data_cfg.id_col: df_test[data_cfg.id_col],
            data_cfg.target_col: test_pred,
        }
    )
    return submission