"""Tests for hierarchical reconciliation: store forecasts sum to state/total."""

import pytest
import numpy as np
from src.models.reconciliation import HierarchicalReconciler


def test_hierarchical_reconciliation_coherency():
    rec = HierarchicalReconciler(method="mint_shrink")
    h = 7
    base_forecasts = {}
    for st in HierarchicalReconciler.BOTTOM_NODES:
        base_forecasts[st] = np.full(h, 10.0)

    # Incoherent totals
    base_forecasts["CA"] = np.full(h, 48.0)
    base_forecasts["TX"] = np.full(h, 28.0)
    base_forecasts["WI"] = np.full(h, 35.0)
    base_forecasts["Total"] = np.full(h, 125.0)

    # Check errors exist before
    pre_err = rec.compute_coherency_error(base_forecasts)
    assert pre_err["Total_vs_States_MaxDiff"] > 0

    # Reconcile
    reconciled = rec.reconcile(base_forecasts)
    post_err = rec.compute_coherency_error(reconciled)

    # Coherency test: store forecasts sum to state and total
    assert np.isclose(post_err["Total_vs_States_MaxDiff"], 0.0, atol=1e-5)
    assert np.isclose(post_err["CA_vs_Stores_MaxDiff"], 0.0, atol=1e-5)
