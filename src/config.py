# src/config.py
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_col: str = "Survived"
    id_col: str = "PassengerId"


@dataclass
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class ModelConfig:
    random_state: int = 42
    cv_folds: int = 5


data_cfg = DataConfig()
split_cfg = SplitConfig()
model_cfg = ModelConfig()