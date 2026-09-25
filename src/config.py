import os
from pathlib import Path
import torch

# Base working directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Detect Kaggle environment
IS_KAGGLE = Path("/kaggle").exists()

# Writable directories for models and outputs
if IS_KAGGLE:
    WORKING_DIR = Path("/kaggle/working")
    MODELS_DIR = WORKING_DIR / "models"
    OUTPUT_DIR = WORKING_DIR / "output"
else:
    WORKING_DIR = BASE_DIR
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "output"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def locate_dataset_paths():
    """
    Intelligently locates train and test directories:
    1. Checks environment variables DATASET_DIR, TRAIN_DIR, TEST_DIR
    2. Searches /kaggle/input/ recursively if running on Kaggle
    3. Falls back to local dataset/train and dataset/test
    """
    env_train = os.environ.get("TRAIN_DIR")
    env_test = os.environ.get("TEST_DIR")
    if env_train and env_test:
        return Path(env_train), Path(env_test)

    env_data = os.environ.get("DATASET_DIR")
    if env_data:
        p = Path(env_data)
        return p / "train", p / "test"

    if IS_KAGGLE:
        kaggle_input = Path("/kaggle/input")
        # Search for train_source1.tsv anywhere in /kaggle/input
        for found_train in kaggle_input.rglob("train_source1.tsv"):
            train_dir = found_train.parent
            # Look for test_source1.tsv in sibling directory or adjacent
            for found_test in kaggle_input.rglob("test_source1.tsv"):
                test_dir = found_test.parent
                return train_dir, test_dir
            # If train has both
            if (train_dir / "test_source1.tsv").exists():
                return train_dir, train_dir
            return train_dir, train_dir.parent / "test"

    local_dataset = BASE_DIR / "dataset"
    return local_dataset / "train", local_dataset / "test"


TRAIN_DIR, TEST_DIR = locate_dataset_paths()
DATASET_DIR = TRAIN_DIR.parent

# Hardware config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multilingual embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Higher batch size on GPU for speed
EMBEDDING_BATCH_SIZE = 256 if torch.cuda.is_available() else 64

# Blocking parameters
SPARSE_TOP_K = 20
DENSE_TOP_K = 15
MAX_UNION_CANDIDATES = 30

# Feature engineering
USE_EMBEDDINGS = True

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 600,
    "verbose": -1,
    "n_jobs": -1
}

# Validation & Decision Threshold Sweep
VAL_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.82
THRESHOLD_SWEEP_RANGE = [round(x * 0.01, 2) for x in range(50, 96)]
