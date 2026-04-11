"""
features.py
───────────
Feature engineering pipeline for neurofeedback data.

Builds the state vector used by both the supervised predictor
and the RL agent.

NOTE: Column names (SIGNAL_COLS, ACTION_COL) are set to placeholders.
      Update them after running 01_eda.ipynb.
"""

import numpy as np
import pandas as pd

TARGET_COL    = "ProtocolValue"
SUBSESSION_COL = "subsession"

# ── TO BE UPDATED AFTER EDA ───────────────────────────────────────────────────
# Fill these after inspecting 01_eda.ipynb outputs
SIGNAL_COLS   = []   # e.g. ["Alpha", "Beta", "Theta", "PlayerPositionY", ...]
ACTION_COL    = None  # e.g. "ThresholdParam" — the controllable system parameter
# ─────────────────────────────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    lags: list[int] = [1, 2, 5],
    rolling_windows: list[int] = [5, 10, 20],
    normalise_to_baseline: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw combined DataFrame.

    Steps:
    1. Lag features for ProtocolValue and signal columns
    2. Rolling statistics (mean, std) within subsession
    3. Delta features (first differences)
    4. Subsession context (index, is_baseline flag)
    5. Optional: z-score normalisation relative to subsession 0 baseline

    Parameters
    ----------
    df                    : combined DataFrame from data_loader
    lags                  : list of lag steps to add for target + signals
    rolling_windows       : list of window sizes for rolling stats
    normalise_to_baseline : if True, z-score features relative to subsession 0

    Returns
    -------
    pd.DataFrame with features + target column + subsession column
    """
    out = df.copy().sort_values([SUBSESSION_COL, "sample_idx"]).reset_index(drop=True)

    # ── 1. Lag features ───────────────────────────────────────────────────────
    cols_to_lag = [TARGET_COL] + [c for c in SIGNAL_COLS if c in out.columns]
    for col in cols_to_lag:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(SUBSESSION_COL)[col].shift(lag)

    # ── 2. Rolling statistics ─────────────────────────────────────────────────
    for col in cols_to_lag:
        for w in rolling_windows:
            out[f"{col}_rmean{w}"] = (
                out.groupby(SUBSESSION_COL)[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            out[f"{col}_rstd{w}"] = (
                out.groupby(SUBSESSION_COL)[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
            )

    # ── 3. Delta features ─────────────────────────────────────────────────────
    out[f"{TARGET_COL}_delta1"] = out.groupby(SUBSESSION_COL)[TARGET_COL].diff(1)
    out[f"{TARGET_COL}_delta2"] = out.groupby(SUBSESSION_COL)[TARGET_COL].diff(2)

    # ── 4. Context features ───────────────────────────────────────────────────
    out["is_baseline"]        = (out[SUBSESSION_COL] == 0).astype(int)
    out["subsession_norm"]    = out[SUBSESSION_COL] / out[SUBSESSION_COL].max()
    out["sample_idx_norm"]    = (
        out.groupby(SUBSESSION_COL)["sample_idx"]
        .transform(lambda x: x / x.max() if x.max() > 0 else x)
    )

    # ── 5. Target: next-step ProtocolValue ────────────────────────────────────
    out[f"{TARGET_COL}_next"] = out.groupby(SUBSESSION_COL)[TARGET_COL].shift(-1)

    # ── 6. Baseline normalisation ─────────────────────────────────────────────
    if normalise_to_baseline and len(SIGNAL_COLS) > 0:
        baseline = df[df[SUBSESSION_COL] == 0]
        for col in SIGNAL_COLS:
            if col not in df.columns:
                continue
            mu  = baseline[col].mean()
            std = baseline[col].std()
            if std > 0:
                out[f"{col}_znorm"] = (out[col] - mu) / std

    return out


def get_feature_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """
    Return the list of feature columns (everything except target and metadata).
    """
    exclude = exclude or []
    non_feature = {
        TARGET_COL,
        f"{TARGET_COL}_next",
        SUBSESSION_COL,
        "sample_idx",
        "session_type",
    } | set(exclude)
    return [c for c in df.columns if c not in non_feature and df[c].dtype != object]
