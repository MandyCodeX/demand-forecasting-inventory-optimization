"""LightGBM Multi-Step Demand Forecasting & Quantile Regression.

Implements Gradient Boosted Decision Trees with Tweedie loss for point forecasts
and Pinball Quantile loss for predictive uncertainty intervals (P10, P50, P90).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except (ImportError, OSError):
    LGBM_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingRegressor
    logger.warning("LightGBM not available. Using HistGradientBoostingRegressor fallback.")


class LightGBMForecaster:
    """Tweedie & Quantile Gradient Boosting for retail demand forecasting."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        quantiles: Optional[List[float]] = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.random_state = random_state
        self.point_model = None
        self.quantile_models: Dict[float, any] = {}
        self.feature_names: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "LightGBMForecaster":
        """Fit point regressor (Tweedie) and quantile regressors (Pinball loss)."""
        self.feature_names = feature_cols or list(X_train.columns)
        X_tr = X_train[self.feature_names]
        y_tr = np.asarray(y_train, dtype=np.float32)

        if LGBM_AVAILABLE:
            # 1. Main point prediction model with Tweedie objective
            params = {
                "objective": "tweedie",
                "tweedie_variance_power": 1.15,
                "metric": "rmse",
                "boosting_type": "gbdt",
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }
            self.point_model = lgb.LGBMRegressor(**params)
            eval_set = [(X_val[self.feature_names], np.asarray(y_val, dtype=np.float32))] if X_val is not None and y_val is not None else None
            
            callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set is not None else []
            self.point_model.fit(
                X_tr,
                y_tr,
                eval_set=eval_set,
                callbacks=callbacks,
            )

            # Store feature importances
            importances = self.point_model.feature_importances_
            self.feature_importances_ = dict(
                sorted(
                    zip(self.feature_names, [float(v) for v in importances]),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

            # 2. Quantile models for predictive intervals
            for q in self.quantiles:
                q_params = {
                    "objective": "quantile",
                    "alpha": q,
                    "boosting_type": "gbdt",
                    "n_estimators": max(100, self.n_estimators // 2),
                    "learning_rate": self.learning_rate,
                    "num_leaves": self.num_leaves,
                    "random_state": self.random_state,
                    "verbose": -1,
                    "n_jobs": -1,
                }
                q_model = lgb.LGBMRegressor(**q_params)
                q_model.fit(X_tr, y_tr)
                self.quantile_models[q] = q_model

        else:
            # Fallback for systems lacking compiled LightGBM
            self.point_model = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            self.point_model.fit(X_tr, y_tr)

            for q in self.quantiles:
                q_model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=max(80, self.n_estimators // 3),
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
                q_model.fit(X_tr, y_tr)
                self.quantile_models[q] = q_model

        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict point demand and quantile bands."""
        X_eval = X[self.feature_names]
        point = np.maximum(0.0, self.point_model.predict(X_eval))

        results = {
            "point": point,
            "q_0.50": point,
        }

        for q, model in self.quantile_models.items():
            key = f"q_{q:.2f}"
            q_pred = np.maximum(0.0, model.predict(X_eval))
            results[key] = q_pred

        # Enforce non-crossing quantile property
        if "q_0.10" in results and "q_0.90" in results:
            results["q_0.10"] = np.minimum(results["q_0.10"], results["q_0.50"])
            results["q_0.90"] = np.maximum(results["q_0.90"], results["q_0.50"])

        return results
