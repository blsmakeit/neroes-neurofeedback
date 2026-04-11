"""
baselines.py
────────────
Naive baselines for comparison against the main prediction and RL modules.

Baselines implemented:
  1. LastValue    — predict ProtocolValue(t+1) = ProtocolValue(t)
  2. RollingMean  — predict using rolling mean of last N samples
  3. SessionMean  — predict using the mean of subsession 0 (baseline)
  4. RandomAction — recommend a random action (lower bound for RL)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COL     = "ProtocolValue"
SUBSESSION_COL = "subsession"


class LastValueBaseline:
    """Predict next value = current value (persistence model)."""

    name = "LastValue"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[TARGET_COL].values


class RollingMeanBaseline:
    """Predict next value = rolling mean of last `window` samples."""

    def __init__(self, window: int = 10):
        self.window = window
        self.name   = f"RollingMean(w={window})"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (
            df.groupby(SUBSESSION_COL)[TARGET_COL]
            .transform(lambda x: x.shift(1).rolling(self.window, min_periods=1).mean())
            .values
        )


class SessionMeanBaseline:
    """Predict next value = mean of subsession 0 (user's personal baseline)."""

    name = "SessionMean(baseline)"

    def fit(self, df: pd.DataFrame) -> "SessionMeanBaseline":
        self.baseline_mean_ = df[df[SUBSESSION_COL] == 0][TARGET_COL].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.baseline_mean_)


class RandomActionBaseline:
    """Recommend a random action from the action space (RL lower bound)."""

    name = "RandomAction"

    def __init__(self, action_space: list, seed: int = 42):
        self.action_space = action_space
        self.rng          = np.random.default_rng(seed)

    def recommend(self, n: int = 1) -> list:
        return self.rng.choice(self.action_space, size=n).tolist()


# ── Evaluation helper ──────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name:   str = "model",
) -> dict:
    """Compute regression metrics between true and predicted values."""
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return {
        "model": name,
        "MAE":   mean_absolute_error(y_true, y_pred),
        "RMSE":  mean_squared_error(y_true, y_pred) ** 0.5,
        "R2":    r2_score(y_true, y_pred),
        "n":     len(y_true),
    }


def compare_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Run all baselines and return a comparison table."""
    target_next = df.groupby(SUBSESSION_COL)[TARGET_COL].shift(-1)
    y_true      = target_next.values

    baselines = [
        LastValueBaseline(),
        RollingMeanBaseline(window=5),
        RollingMeanBaseline(window=10),
        RollingMeanBaseline(window=20),
        SessionMeanBaseline().fit(df),
    ]

    results = []
    for bl in baselines:
        y_pred  = bl.predict(df)
        metrics = evaluate_predictions(y_true, y_pred, name=bl.name)
        results.append(metrics)

    return pd.DataFrame(results).set_index("model").sort_values("MAE")
