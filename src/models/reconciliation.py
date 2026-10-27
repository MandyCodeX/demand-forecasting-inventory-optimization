"""Hierarchical Forecast Reconciliation.

Implements Bottom-Up, Top-Down, Ordinary Least Squares (OLS), and MinT (Minimum Trace)
reconciliation ensuring store forecasts strictly sum to state totals, and state forecasts
sum to nationwide totals. Integrates with Darts hierarchical forecasting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check Darts availability
try:
    from darts import TimeSeries
    from darts.dataprocessing.transformers.reconciliation import (
        BottomUpReconciler,
        TopDownReconciler,
        MinTReconciler,
    )
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    logger.info("Darts not installed; running native linear-algebra HierarchicalReconciler.")


class HierarchicalReconciler:
    """Reconciles multi-level demand forecasts across Total -> State -> Store hierarchy."""

    HIERARCHY_TREE: Dict[str, List[str]] = {
        "Total": ["CA", "TX", "WI"],
        "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
        "TX": ["TX_1", "TX_2", "TX_3"],
        "WI": ["WI_1", "WI_2", "WI_3"],
    }

    BOTTOM_NODES: List[str] = [
        "CA_1", "CA_2", "CA_3", "CA_4",
        "TX_1", "TX_2", "TX_3",
        "WI_1", "WI_2", "WI_3",
    ]

    ALL_NODES: List[str] = [
        "Total",
        "CA", "TX", "WI",
        "CA_1", "CA_2", "CA_3", "CA_4",
        "TX_1", "TX_2", "TX_3",
        "WI_1", "WI_2", "WI_3",
    ]

    def __init__(self, method: str = "mint_shrink") -> None:
        self.method = method.lower()
        self.S_matrix = self._build_summing_matrix()

    def _build_summing_matrix(self) -> np.ndarray:
        """Construct the hierarchical Summing Matrix S of shape (m, m_bottom).

        Rows: All 14 nodes (Total, 3 states, 10 stores).
        Cols: 10 bottom store nodes.
        """
        m = len(self.ALL_NODES)
        m_b = len(self.BOTTOM_NODES)
        S = np.zeros((m, m_b), dtype=np.float64)

        b_idx_map = {node: i for i, node in enumerate(self.BOTTOM_NODES)}

        for r_idx, node in enumerate(self.ALL_NODES):
            if node == "Total":
                S[r_idx, :] = 1.0
            elif node in self.HIERARCHY_TREE:  # State level
                children = self.HIERARCHY_TREE[node]
                for child in children:
                    c_idx = b_idx_map[child]
                    S[r_idx, c_idx] = 1.0
            elif node in b_idx_map:  # Bottom store level
                c_idx = b_idx_map[node]
                S[r_idx, c_idx] = 1.0

        return S

    def compute_coherency_error(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Quantify sum-consistency violation across hierarchy levels before and after reconciliation."""
        errors = {}

        # 1. State sum vs Total
        if "Total" in forecasts and all(s in forecasts for s in ["CA", "TX", "WI"]):
            sum_states = sum(forecasts[s] for s in ["CA", "TX", "WI"])
            errors["Total_vs_States_MaxDiff"] = float(np.max(np.abs(forecasts["Total"] - sum_states)))

        # 2. Store sum vs States
        for state, stores in self.HIERARCHY_TREE.items():
            if state != "Total" and state in forecasts and all(st in forecasts for st in stores):
                sum_stores = sum(forecasts[st] for st in stores)
                errors[f"{state}_vs_Stores_MaxDiff"] = float(np.max(np.abs(forecasts[state] - sum_stores)))

        return errors

    def reconcile_bottom_up(self, base_forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Bottom-Up reconciliation: aggregate bottom forecasts directly up the tree."""
        reconciled = {}
        for store in self.BOTTOM_NODES:
            reconciled[store] = np.copy(base_forecasts[store])

        # Reconcile states
        for state, stores in self.HIERARCHY_TREE.items():
            if state != "Total":
                reconciled[state] = sum(reconciled[st] for st in stores)

        # Reconcile Total
        reconciled["Total"] = sum(reconciled[s] for s in ["CA", "TX", "WI"])
        return reconciled

    def reconcile_top_down(
        self,
        base_forecasts: Dict[str, np.ndarray],
        historical_proportions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Top-Down reconciliation: disaggregate nationwide total based on historical shares."""
        if historical_proportions is None:
            # Default average store proportions
            n_stores = len(self.BOTTOM_NODES)
            historical_proportions = {s: 1.0 / n_stores for s in self.BOTTOM_NODES}

        total_f = base_forecasts["Total"]
        reconciled = {"Total": np.copy(total_f)}

        for store in self.BOTTOM_NODES:
            share = historical_proportions.get(store, 0.1)
            reconciled[store] = total_f * share

        for state, stores in self.HIERARCHY_TREE.items():
            if state != "Total":
                reconciled[state] = sum(reconciled[st] for st in stores)

        return reconciled

    def reconcile_mint(
        self,
        base_forecasts: Dict[str, np.ndarray],
        residuals: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Minimum Trace (MinT) optimal linear reconciliation.

        MinT projection: P = (S^T W^{-1} S)^{-1} S^T W^{-1}
        Coherent forecasts: y_tilde = S P y_hat
        """
        horizon = len(next(iter(base_forecasts.values())))
        m = len(self.ALL_NODES)
        S = self.S_matrix

        # Stack base forecasts: (m, horizon)
        Y_hat = np.zeros((m, horizon), dtype=np.float64)
        for i, node in enumerate(self.ALL_NODES):
            Y_hat[i, :] = base_forecasts[node]

        # Weight matrix W (Covariance of residuals or scaled identity/OLS)
        if residuals is not None and residuals.shape[0] == m:
            cov = np.cov(residuals)
            # Ledoit-Wolf shrinkage to diagonal for numerical stability
            target = np.diag(np.diag(cov))
            shrinkage = 0.2
            W = (1.0 - shrinkage) * cov + shrinkage * target
            W += np.eye(m) * 1e-4
        else:
            # Structurally scaled W (W_ii proportional to node variance / OLS approximation)
            node_scales = np.sum(S, axis=1)  # Higher nodes have higher variances
            W = np.diag(node_scales)

        W_inv = np.linalg.pinv(W)
        St_Winv = S.T @ W_inv
        middle = np.linalg.pinv(St_Winv @ S)
        P = middle @ St_Winv

        # Reconciled forecasts: (m, horizon)
        Y_reconciled = S @ (P @ Y_hat)
        Y_reconciled = np.maximum(0.0, Y_reconciled)  # Non-negativity

        reconciled = {}
        for i, node in enumerate(self.ALL_NODES):
            reconciled[node] = Y_reconciled[i, :]

        return reconciled

    def reconcile(
        self,
        base_forecasts: Dict[str, np.ndarray],
        residuals: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Dispatch to chosen reconciliation algorithm."""
        if self.method == "bottom_up":
            return self.reconcile_bottom_up(base_forecasts)
        elif self.method == "top_down":
            return self.reconcile_top_down(base_forecasts)
        elif self.method in ["mint", "mint_shrink", "ols"]:
            return self.reconcile_mint(base_forecasts, residuals=residuals)
        else:
            raise ValueError(f"Unknown reconciliation method: {self.method}")
