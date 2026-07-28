"""Load and clean athletes.csv for AutoML."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_and_clean(csv_path=None) -> pd.DataFrame:
    """Load the raw CSV and return a cleaned athletes DataFrame."""
    path = csv_path or config.RAW_CSV
    df = pd.read_csv(path)

    # Limit to relevant columns
    df = df[config.KEEP_COLUMNS].copy()

    # Remove outliers
    df = df[df["weight"] < 1500]
    df = df[df["gender"] != "--"]
    df = df[df["age"] >= 18]
    df = df[(df["height"] < 96) & (df["height"] > 48)]

    df = df[
        ((df["gender"] == "Male") & (df["deadlift"] <= 1105))
        | ((df["gender"] == "Female") & (df["deadlift"] <= 636))
    ]
    df = df[(df["candj"] > 0) & (df["candj"] <= 395)]
    df = df[(df["snatch"] > 0) & (df["snatch"] <= 496)]
    df = df[(df["backsq"] > 0) & (df["backsq"] <= 1069)]

    # Clean survey data
    decline_dict = {"Decline to answer|": np.nan}
    df = df.replace(decline_dict)

    # Remove missing values
    df = df.dropna().copy()

    df[config.TARGET] = df[config.LIFT_COMPONENTS].sum(axis=1)

    return df.reset_index(drop=True)


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric encodings used by ALL_FEATURES / BASELINE_FEATURES."""
    out = df.copy()
    out["gender_male"] = (out["gender"] == "Male").astype("int64")
    howlong = out["howlong"].fillna("").astype(str).str.lower()
    out["is_experienced"] = howlong.str.contains(
        "2-4 years|4\\+ years", regex=True
    ).astype("int64")
    background = out["background"].fillna("").astype(str).str.lower()
    out["has_athletic_background"] = (
        ~background.str.contains("no athletic background", regex=False)
    ).astype("int64")
    return out


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_cols) with leakage columns excluded.

    Encodes gender and survey flags. Drops lift components and the target
    from predictors even if they appear in ``feature_cols``.
    """
    encoded = _encode_features(df)
    cols = list(feature_cols) if feature_cols is not None else list(config.ALL_FEATURES)

    banned = set(config.LIFT_COMPONENTS) | {config.TARGET, "gender", "howlong", "background"}
    cols = [c for c in cols if c not in banned]

    missing = [c for c in cols if c not in encoded.columns]
    if missing:
        raise KeyError(f"Requested features not in frame after encoding: {missing}")

    X = encoded[cols].copy()
    y = encoded[config.TARGET].copy()
    if X.isna().any().any():
        raise ValueError("Feature matrix still contains NaNs after cleaning/encoding")
    return X, y, cols


def train_test_split_xy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float | None = None,
    random_state: int | None = None,
):
    """80/20 train/test split with Assignment seed defaults."""
    return train_test_split(
        X,
        y,
        test_size=test_size if test_size is not None else config.TEST_SIZE,
        random_state=random_state if random_state is not None else config.SEED,
    )


def save_clean_parquet(df: pd.DataFrame | None = None, path=None) -> Path:
    """Write cleaned frame to parquet; return the path."""
    out = Path(path) if path is not None else config.CLEAN_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = df if df is not None else load_and_clean()
    frame.to_parquet(out, index=False)
    return out


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(config.RAW_CSV)
    clean = load_and_clean()
    out = save_clean_parquet(clean)

    X, y, cols = get_feature_matrix(clean)
    print(f"raw rows:     {len(raw):,}")
    print(f"cleaned rows: {len(clean):,}")
    print(f"target:       {config.TARGET}")
    print(f"features:     {cols}")
    print(f"wrote:        {out}")
    print(f"X shape:      {X.shape}, y nulls: {int(y.isna().sum())}")


if __name__ == "__main__":
    main()
