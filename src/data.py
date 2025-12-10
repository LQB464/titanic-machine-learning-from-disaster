# src/data.py
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, data_cfg, split_cfg


def load_raw_data() -> pd.DataFrame:
    """Load train.csv gốc."""
    path = DATA_DIR / data_cfg.train_file
    df = pd.read_csv(path)
    return df


def load_test_data() -> pd.DataFrame:
    path = DATA_DIR / data_cfg.test_file
    df = pd.read_csv(path)
    return df


def basic_checks(df: pd.DataFrame) -> None:
    """Một số check nhẹ, không vẽ biểu đồ (EDA vẫn ở notebook)."""
    print("Shape:", df.shape)
    print("Dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())


def split_train_valid(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dữ liệu train thành train/valid."""
    if target_col is None:
        target_col = data_cfg.target_col

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if split_cfg.stratify else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        stratify=stratify,
    )
    return X_train, X_valid, y_train, y_valid