# src/features.py
from typing import Tuple

import numpy as np
import pandas as pd


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Title từ Name."""
    df = df.copy()
    df["Title"] = (
        df["Name"]
        .str.split(",", expand=True)[1]
        .str.split(".", expand=True)[0]
        .str.strip()
    )
    return df


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """FamilySize = SibSp + Parch + 1, thêm bucket."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    def bucket(n: int) -> str:
        if n == 1:
            return "Alone"
        if 2 <= n <= 4:
            return "Medium"
        return "Large"

    df["FamilySize_Bucketized"] = df["FamilySize"].apply(bucket)
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_cabin_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def add_fare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fare per passenger = Fare / FamilySize
    if "FamilySize" not in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["FarePerPassenger"] = df["Fare"] / df["FamilySize"]
    df["LogFare"] = np.log1p(df["Fare"])
    return df


def fill_basic_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Một số xử lý NA giống trong notebook (ví dụ Embarked)."""
    df = df.copy()

    # Embarked: thay NA bằng mode hoặc "S"
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Age / Fare sẽ được xử lý trong pipeline bằng Imputer nên không fill ở đây
    return df


def full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Gom tất cả bước feature engineering chính."""
    df = fill_basic_missing(df)
    df = extract_title(df)
    df = add_family_size(df)
    df = add_cabin_flag(df)
    df = add_fare_features(df)
    return df