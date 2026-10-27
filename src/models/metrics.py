"""Forecast Accuracy & Uncertainty Evaluation Metrics.

Includes WAPE (Weighted Absolute Percentage Error) and RMSSE (Root Mean Squared
Scaled Error), the official primary metric used in the Walmart M5 Forecasting competition.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|y - y_hat|) / sum(y)
    Avoids division by zero on intermittent zero-demand days.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    sum_true = np.sum(np.abs(y_t))
    if sum_true == 0:
        return 0.0 if np.sum(np.abs(y_p)) == 0 else 1.0

    return float(np.sum(np.abs(y_t - y_p)) / sum_true)


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray,
) -> float:
    """Root Mean Squared Scaled Error (RMSSE - M5 Official Metric).

    Scales forecast MSE by in-sample one-step naive forecast MSE.
    RMSSE = sqrt( (1/h * sum((y_t - y_hat_t)^2)) / (1/(n-1) * sum((y_i - y_{i-1})^2)) )
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_hist = np.asarray(y_train_history, dtype=np.float64)

    h = len(y_t)
    if h == 0:
        return 0.0

    # Numerator: MSE of forecast horizon
    numerator = np.mean((y_t - y_p) ** 2)

    # Denominator: In-sample naive differences squared
    diffs = np.diff(y_hist)
    if len(diffs) == 0:
        return float(np.sqrt(numerator))

    scale = np.mean(diffs ** 2)
    if scale == 0:
        scale = 1e-4

    return float(np.sqrt(numerator / scale))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def interval_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Empirical coverage rate: proportion of true values falling within [lower, upper]."""
    y_t = np.asarray(y_true)
    y_l = np.asarray(y_lower)
    y_u = np.asarray(y_upper)

    within = (y_t >= y_l) & (y_t <= y_u)
    return float(np.mean(within))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive performance summary metrics."""
    results = {
        "WAPE": round(wape(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
    }

    if y_train_history is not None:
        results["RMSSE"] = round(rmsse(y_true, y_pred, y_train_history), 4)

    if y_lower is not None and y_upper is not None:
        results["Coverage_80"] = round(interval_coverage(y_true, y_lower, y_upper), 4)

    return results
