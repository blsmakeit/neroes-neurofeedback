"""
evaluation.py
─────────────
Evaluation utilities for both prediction and recommendation modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COL = "ProtocolValue"

PLOT_STYLE = {
    "bg":     "#0d1117",
    "surface":"#161b22",
    "cyan":   "#00c8ff",
    "orange": "#ff6b35",
    "teal":   "#00e5c3",
    "white":  "#e6edf3",
    "muted":  "#8b949e",
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "MAE":  round(mean_absolute_error(yt, yp), 6),
        "RMSE": round(mean_squared_error(yt, yp) ** 0.5, 6),
        "R2":   round(r2_score(yt, yp), 6),
        "n":    int(mask.sum()),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of steps where predicted direction (up/down) matches true direction."""
    delta_true = np.diff(y_true)
    delta_pred = np.diff(y_pred)
    mask = ~(np.isnan(delta_true) | np.isnan(delta_pred))
    return float(np.mean(np.sign(delta_true[mask]) == np.sign(delta_pred[mask])))


def improvement_rate(
    df: pd.DataFrame,
    recommended_col: str,
    target_col: str = TARGET_COL,
    subsession_col: str = "subsession",
) -> float:
    """
    Fraction of recommended actions that lead to a positive Δ ProtocolValue.
    Used to evaluate recommendation quality.
    """
    game = df[df[subsession_col] > 0].copy()
    game["delta"] = game.groupby(subsession_col)[target_col].diff(1)
    positive = (game["delta"] > 0).mean()
    return float(positive)


def plot_prediction_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title:  str = "Prediction vs. Ground Truth",
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(PLOT_STYLE["bg"])
    fig.suptitle(title, color=PLOT_STYLE["cyan"], fontsize=13, fontweight="bold")

    n = min(500, len(y_true))
    axes[0].plot(y_true[:n], color=PLOT_STYLE["white"], lw=1.2, label="True", alpha=0.9)
    axes[0].plot(y_pred[:n], color=PLOT_STYLE["orange"], lw=1.2, label="Predicted", alpha=0.8)
    axes[0].set_facecolor(PLOT_STYLE["surface"])
    axes[0].set_title("Time Series (first 500 samples)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(y_true, y_pred, alpha=0.2, s=3,
                    color=PLOT_STYLE["cyan"], edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, color=PLOT_STYLE["orange"], lw=1.5, linestyle="--", label="Perfect")
    axes[1].set_facecolor(PLOT_STYLE["surface"])
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Scatter (True vs. Predicted)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PLOT_STYLE["bg"])
    plt.show()


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Build a formatted comparison table from a list of metric dicts.
    Each dict must have keys: model, MAE, RMSE, R2
    """
    df = pd.DataFrame(results).set_index("model")
    df = df.sort_values("MAE")
    df["rank"] = range(1, len(df) + 1)
    return df
