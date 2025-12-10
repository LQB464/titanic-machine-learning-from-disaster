# src/preprocessing.py
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUM_COLS = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "FarePerPassenger", "LogFare"]
CAT_COLS = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "FamilySize_Bucketized",
    "HasCabin",
    "IsAlone",
]


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Tạo ColumnTransformer gần giống trong notebook."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUM_COLS),
            ("cat", categorical_pipeline, CAT_COLS),
        ]
    )

    return preprocessor


def fit_preprocess(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> np.ndarray:
    return preprocessor.fit_transform(X_train)


def transform_preprocess(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    return preprocessor.transform(X)