"""Paths, constants, and AutoML runtime knobs for Assignment 3."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "athletes.csv"
CLEAN_PARQUET = DATA_DIR / "athletes_clean.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "athlete_total_lift_automl"

# Reproducibility
SEED = 42
TEST_SIZE = 0.2

# Target definition
TARGET = "total_lift"
LIFT_COMPONENTS = ["deadlift", "candj", "snatch", "backsq"]

# Columns kept from the raw CSV during cleaning
KEEP_COLUMNS = [
    "athlete_id",
    "gender",
    "age",
    "height",
    "weight",
    "howlong",
    "background",
    *LIFT_COMPONENTS,
]

# Default modeling features after cleaning / encoding
ALL_FEATURES = [
    "age",
    "height",
    "weight",
    "gender_male",
    "is_experienced",
    "has_athletic_background",
]

# Assign 1 baseline feature set
BASELINE_FEATURES = ["age", "height", "weight", "gender_male"]

# TPOT runtime knobs
TPOT_MAX_TIME_MINS = 10
TPOT_GENERATIONS = 5
TPOT_POPULATION_SIZE = 20
TPOT_CV = 5
TPOT_N_JOBS = -1

# H2O AutoML knobs
H2O_MAX_RUNTIME_SECS = 180
H2O_MAX_MODELS = 40
H2O_NFOLDS = 5
