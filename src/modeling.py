# src/modeling.py
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .preprocessing import build_preprocessing_pipeline


def build_base_models() -> Dict[str, Any]:
    """Các model baseline giống phần Approach 1 trong notebook."""
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "rf": RandomForestClassifier(),
        "svc": SVC(probability=True),
        "xgb": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        ),
    }
    return models


def build_pipeline_for_model(model) -> Pipeline:
    """Gói preprocessor + model vào một Pipeline."""
    preprocessor = build_preprocessing_pipeline()
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def build_pca_pipeline(model, n_components: int = 10) -> Pipeline:
    """PCA trên feature đã preprocess, giống phần PCA trong notebook."""
    preprocessor = build_preprocessing_pipeline()
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("pca", PCA(n_components=n_components)),
            ("model", model),
        ]
    )
    return pipe


def build_voting_classifier() -> VotingClassifier:
    """Ensemble KNN, RF, XGB giống phần Voting Classifier."""
    base_models = build_base_models()

    estimators = [
        ("knn", base_models["knn"]),
        ("rf", base_models["rf"]),
        ("xgb", base_models["xgb"]),
    ]

    voting_clf = VotingClassifier(
        estimators=estimators,
        voting="soft",
    )
    return voting_clf


def get_knn_param_grid() -> Dict[str, List[Any]]:
    """Grid cho KNN, tương ứng phần hyperparameter tuning model tốt nhất."""
    return {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }