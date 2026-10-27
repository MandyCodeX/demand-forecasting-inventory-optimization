"""Baseline Forecasting Models.

Provides fast, benchmark implementations of:
- Naive (Last Known Value)
- Seasonal Naive (Weekly / 7-Day Cycle)
- Moving Average (Rolling Window with Parametric Prediction Intervals)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class NaiveModel:
    """Predicts the most recently observed value for all future time steps."""

    def __init__(self) -> None:
        self.last_value: float = 0.0
        self.residuals_std: float = 1.0

    def fit(self, y: np.ndarray | pd.Series) -> "NaiveModel":
        arr = np.asarray(y, dtype=np.float64)
        self.last_value = float(arr[-1]) if len(arr) > 0 else 0.0
        if len(arr) > 1:
            diffs = np.diff(arr)
            self.residuals_std = float(np.std(diffs)) or 1.0
        return self

    def predict(
        self, horizon: int, quantiles: Optional[Tuple[float, ...]] = (0.1, 0.5, 0.9)
    ) -> Dict[str, np.ndarray]:
        point = np.full(horizon, self.last_value, dtype=np.float64)
        results = {"q_0.50": point, "point": point}

        # Step-dependent variance: sigma_t = sigma * sqrt(t)
        t_steps = np.arange(1, horizon + 1)
        sigma_t = self.residuals_std * np.sqrt(t_steps)

        # Normal approximation quantiles: z_0.10 = -1.282, z_0.90 = +1.282
        results["q_0.10"] = np.maximum(0.0, point - 1.282 * sigma_t)
        results["q_0.90"] = point + 1.282 * sigma_t
        return results


class SeasonalNaiveModel:
    """Repeats historical seasonal pattern (default season = 7 days for weekly retail cycles)."""

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length
        self.seasonal_pattern: np.ndarray = np.array([])
        self.residuals_std: float = 1.0

    def fit(self, y: np.ndarray | pd.Series) -> "SeasonalNaiveModel":
        arr = np.asarray(y, dtype=np.float64)
        if len(arr) < self.season_length:
            self.seasonal_pattern = np.resize(arr, self.season_length)
        else:
            self.seasonal_pattern = arr[-self.season_length :]

        if len(arr) > self.season_length:
            season_diffs = arr[self.season_length :] - arr[: -self.season_length]
            self.residuals_std = float(np.std(season_diffs)) or 1.0
        return self

    def predict(
        self, horizon: int, quantiles: Optional[Tuple[float, ...]] = (0.1, 0.5, 0.9)
    ) -> Dict[str, np.ndarray]:
        reps = int(np.ceil(horizon / self.season_length))
        point = np.tile(self.seasonal_pattern, reps)[:horizon]
        results = {"q_0.50": point, "point": point}

        k_seasons = np.ceil(np.arange(1, horizon + 1) / self.season_length)
        sigma_t = self.residuals_std * np.sqrt(k_seasons)

        results["q_0.10"] = np.maximum(0.0, point - 1.282 * sigma_t)
        results["q_0.90"] = point + 1.282 * sigma_t
        return results


class MovingAverageModel:
    """Rolling moving average benchmark over recent window."""

    def __init__(self, window: int = 14) -> None:
        self.window = window
        self.mean_value: float = 0.0
        self.std_value: float = 1.0

    def fit(self, y: np.ndarray | pd.Series) -> "MovingAverageModel":
        arr = np.asarray(y, dtype=np.float64)
        recent = arr[-self.window :] if len(arr) >= self.window else arr
        self.mean_value = float(np.mean(recent)) if len(recent) > 0 else 0.0
        self.std_value = float(np.std(recent)) if len(recent) > 1 else 1.0
        return self

    def predict(
        self, horizon: int, quantiles: Optional[Tuple[float, ...]] = (0.1, 0.5, 0.9)
    ) -> Dict[str, np.ndarray]:
        point = np.full(horizon, self.mean_value, dtype=np.float64)
        results = {"q_0.50": point, "point": point}
        results["q_0.10"] = np.maximum(0.0, point - 1.282 * self.std_value)
        results["q_0.90"] = point + 1.282 * self.std_value
        return results
