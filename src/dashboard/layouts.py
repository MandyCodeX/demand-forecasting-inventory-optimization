"""Dashboard Layouts and Navigation.

Defines the multi-tab responsive UI structure incorporating Dash Bootstrap Components:
- Executive Overview & KPI Dashboard
- Demand Forecasting & Prediction Intervals
- Darts Hierarchical Reconciliation View
- Google OR-Tools Inventory Optimization Workbench
- Model Benchmark & Accuracy Suite (WAPE, RMSSE)
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table


def create_navbar() -> dbc.Navbar:
    """Render the top navigation header bar."""
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Span("📦", className="fs-3 me-2"),
                                width="auto",
                            ),
                            dbc.Col(
                                [
                                    html.H4(
                                        "Nationwide Demand Forecasting & Inventory Optimization",
                                        className="fw-bold mb-0 text-white fs-5",
                                    ),
                                    html.Small(
                                        "Walmart M5 Hierarchical Intelligence Platform | OR-Tools MILP Solver",
                                        className="text-light opacity-75",
                                    ),
                                ],
                                width="auto",
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.Nav(
                    [
                        dbc.Badge("Walmart M5 Dataset", color="primary", className="me-2 p-2"),
                        dbc.Badge("LightGBM + TFT + Prophet", color="info", className="me-2 p-2"),
                        dbc.Badge("OR-Tools MILP", color="success", className="p-2"),
                    ],
                    className="ms-auto d-none d-md-flex align-items-center",
                ),
            ],
            fluid=True,
        ),
        color="#1e293b",
        dark=True,
        className="mb-4 shadow-sm py-2",
    )


def create_filter_bar(
    states: list[str],
    stores: list[str],
    categories: list[str],
    items: list[str],
) -> dbc.Card:
    """Render top hierarchy filter control panel."""
    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Hierarchy Level", className="fw-semibold small text-muted"),
                            dcc.Dropdown(
                                id="dropdown-hierarchy-level",
                                options=[
                                    {"label": "Level 1: Nationwide (Total)", "value": "total"},
                                    {"label": "Level 2: State", "value": "state"},
                                    {"label": "Level 3: Store", "value": "store"},
                                    {"label": "Level 4: Category", "value": "cat"},
                                    {"label": "Level 6: Store x Item (SKU)", "value": "item_store"},
                                ],
                                value="store",
                                clearable=False,
                            ),
                        ],
                        md=3,
                        xs=12,
                    ),
                    dbc.Col(
                        [
                            html.Label("State", className="fw-semibold small text-muted"),
                            dcc.Dropdown(
                                id="dropdown-state",
                                options=[{"label": s, "value": s} for s in states],
                                value=states[0] if states else "CA",
                                clearable=False,
                            ),
                        ],
                        md=2,
                        xs=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Store", className="fw-semibold small text-muted"),
                            dcc.Dropdown(
                                id="dropdown-store",
                                options=[{"label": s, "value": s} for s in stores],
                                value=stores[0] if stores else "CA_1",
                                clearable=False,
                            ),
                        ],
                        md=2,
                        xs=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Category", className="fw-semibold small text-muted"),
                            dcc.Dropdown(
                                id="dropdown-category",
                                options=[{"label": c, "value": c} for c in categories],
                                value=categories[0] if categories else "FOODS",
                                clearable=False,
                            ),
                        ],
                        md=2,
                        xs=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Item SKU", className="fw-semibold small text-muted"),
                            dcc.Dropdown(
                                id="dropdown-item",
                                options=[{"label": i, "value": i} for i in items],
                                value=items[0] if items else "FOODS_1_001",
                                clearable=False,
                            ),
                        ],
                        md=3,
                        xs=6,
                    ),
                ],
                className="g-3 align-items-end",
            )
        ),
        className="shadow-sm mb-4 bg-light border-0",
    )


def create_forecasting_tab() -> dbc.Container:
    """Tab 1: Demand Forecasting & Uncertainty Intervals."""
    return dbc.Container(
        [
            # KPI Metric Banner
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="kpi-projected-demand"),
                        md=3,
                        xs=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        html.Div(id="kpi-accuracy-rmsse"),
                        md=3,
                        xs=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        html.Div(id="kpi-accuracy-wape"),
                        md=3,
                        xs=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        html.Div(id="kpi-order-recommendation"),
                        md=3,
                        xs=6,
                        className="mb-3",
                    ),
                ],
                className="mb-2",
            ),
            # Forecast Plot & Controls
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H5("Multi-Horizon Forecast & Prediction Intervals", className="fw-bold mb-0"),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.RadioItems(
                                        id="radio-forecast-model",
                                        options=[
                                            {"label": "LightGBM (Tweedie)", "value": "LightGBM"},
                                            {"label": "TFT (PyTorch)", "value": "TFT"},
                                            {"label": "Prophet", "value": "Prophet"},
                                            {"label": "Naive Baseline", "value": "Naive"},
                                            {"label": "Reconciled (MinT)", "value": "Reconciled"},
                                        ],
                                        value="LightGBM",
                                        inline=True,
                                        className="text-end",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="align-items-center mb-3",
                        ),
                        dcc.Loading(
                            dcc.Graph(id="chart-demand-forecast", style={"height": "480px"})
                        ),
                        html.Div(
                            html.Small(
                                "ℹ️ Shaded ribbon represents calibrated 80% prediction interval (P10 - P90 quantiles). "
                                "Dotted line shows reconciled coherent forecast across hierarchical aggregation tiers.",
                                className="text-muted",
                            ),
                            className="mt-2",
                        ),
                    ]
                ),
                className="shadow-sm mb-4 border-0",
            ),
        ],
        fluid=True,
    )


def create_reconciliation_tab() -> dbc.Container:
    """Tab 2: Hierarchical Reconciliation (Darts)."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Hierarchical Coherency Principle", className="fw-bold text-dark"),
                                    html.P(
                                        "In retail supply chains, independently forecasting store sales leads to incoherency: "
                                        "the sum of individual store forecasts does not equal direct regional/state or nationwide forecasts. "
                                        "Hierarchical reconciliation ensures mathematical consistency across all tiers.",
                                        className="text-secondary small",
                                    ),
                                    html.Label("Reconciliation Method", className="fw-semibold small text-muted"),
                                    dcc.Dropdown(
                                        id="dropdown-reconciliation-method",
                                        options=[
                                            {"label": "MinT (Minimum Trace with Shrinkage) - Optimal", "value": "mint_shrink"},
                                            {"label": "Bottom-Up (Sum Stores to State & Nation)", "value": "bottom_up"},
                                            {"label": "Top-Down (Disaggregate by Historical Share)", "value": "top_down"},
                                            {"label": "Ordinary Least Squares (OLS)", "value": "ols"},
                                        ],
                                        value="mint_shrink",
                                        clearable=False,
                                        className="mb-3",
                                    ),
                                    html.Div(id="reconciliation-audit-table"),
                                ]
                            ),
                            className="shadow-sm border-0 h-100",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Multi-Level Coherency Comparison", className="fw-bold mb-3"),
                                    dcc.Loading(
                                        dcc.Graph(id="chart-reconciliation", style={"height": "420px"})
                                    ),
                                ]
                            ),
                            className="shadow-sm border-0 h-100",
                        ),
                        md=8,
                    ),
                ],
                className="g-3 mb-4",
            )
        ],
        fluid=True,
    )


def create_inventory_tab() -> dbc.Container:
    """Tab 3: Operations Research Inventory Optimization (Google OR-Tools)."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    # Controls Sidebar
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Supply Chain Constraints", className="fw-bold mb-3 text-dark"),
                                    html.Label("Lead Time (Days)", className="fw-semibold small text-muted"),
                                    dcc.Slider(
                                        id="slider-lead-time",
                                        min=1,
                                        max=7,
                                        step=1,
                                        value=2,
                                        marks={i: str(i) for i in range(1, 8)},
                                    ),
                                    html.Label("Unit Holding Cost ($ / unit / day)", className="fw-semibold small text-muted mt-3"),
                                    dcc.Slider(
                                        id="slider-holding-cost",
                                        min=0.01,
                                        max=0.30,
                                        step=0.01,
                                        value=0.05,
                                        marks={0.01: "$0.01", 0.15: "$0.15", 0.30: "$0.30"},
                                    ),
                                    html.Label("Stockout Penalty Multiplier (Lost Margin)", className="fw-semibold small text-muted mt-3"),
                                    dcc.Slider(
                                        id="slider-stockout-penalty",
                                        min=1.0,
                                        max=6.0,
                                        step=0.5,
                                        value=3.5,
                                        marks={1: "1x", 3.5: "3.5x", 6: "6x"},
                                    ),
                                    html.Label("Warehouse Capacity Limit (Units)", className="fw-semibold small text-muted mt-3"),
                                    dcc.Input(
                                        id="input-warehouse-capacity",
                                        type="number",
                                        value=5000,
                                        className="form-control form-control-sm mb-2",
                                    ),
                                    html.Label("Procurement Daily Budget ($)", className="fw-semibold small text-muted mt-2"),
                                    dcc.Input(
                                        id="input-daily-budget",
                                        type="number",
                                        value=12000,
                                        className="form-control form-control-sm mb-3",
                                    ),
                                    html.Label("Demand Risk Buffer", className="fw-semibold small text-muted"),
                                    dcc.RadioItems(
                                        id="radio-demand-buffer",
                                        options=[
                                            {"label": "Median Expected (P50)", "value": "demand_p50"},
                                            {"label": "Conservative Surge Protection (P90)", "value": "demand_p90"},
                                        ],
                                        value="demand_p50",
                                        className="small mb-4",
                                    ),
                                    dbc.Button(
                                        "⚡ Solve Replenishment MILP",
                                        id="btn-solve-inventory",
                                        color="primary",
                                        className="w-100 fw-bold shadow-sm",
                                    ),
                                ]
                            ),
                            className="shadow-sm border-0 mb-3",
                        ),
                        md=4,
                    ),
                    # Optimization Results & Chart
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(html.Div(id="kpi-opt-total-cost"), md=3, xs=6, className="mb-3"),
                                    dbc.Col(html.Div(id="kpi-opt-service-level"), md=3, xs=6, className="mb-3"),
                                    dbc.Col(html.Div(id="kpi-opt-holding-cost"), md=3, xs=6, className="mb-3"),
                                    dbc.Col(html.Div(id="kpi-opt-savings"), md=3, xs=6, className="mb-3"),
                                ]
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Optimized Multi-Period Inventory Trajectory", className="fw-bold mb-3"),
                                        dcc.Loading(
                                            dcc.Graph(id="chart-inventory-trajectory", style={"height": "380px"})
                                        ),
                                    ]
                                ),
                                className="shadow-sm border-0 mb-4",
                            ),
                        ],
                        md=8,
                    ),
                ],
                className="g-3 mb-4",
            ),
            # Replenishment Order Table
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H5("Automated Replenishment Purchase Orders (OR-Tools Schedule)", className="fw-bold mb-0"),
                                    md=8,
                                ),
                                dbc.Col(
                                    html.Div(
                                        dbc.Button("Download Order CSV", id="btn-export-orders", color="secondary", outline=True, size="sm"),
                                        className="text-end",
                                    ),
                                    md=4,
                                ),
                            ],
                            className="align-items-center mb-3",
                        ),
                        html.Div(id="table-orders-container"),
                    ]
                ),
                className="shadow-sm border-0 mb-4",
            ),
        ],
        fluid=True,
    )


def create_benchmarks_tab() -> dbc.Container:
    """Tab 4: Model Accuracy & Benchmarking (WAPE, RMSSE)."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Accuracy Benchmark by Candidate Model", className="fw-bold mb-3"),
                                    dcc.Graph(id="chart-accuracy-comparison", style={"height": "400px"}),
                                ]
                            ),
                            className="shadow-sm border-0 mb-4",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Official M5 Metric Evaluation Table", className="fw-bold mb-3"),
                                    html.Div(id="table-metrics-benchmark"),
                                    html.Div(
                                        [
                                            html.H6("Metric Formulas:", className="fw-bold mt-4 mb-2"),
                                            html.P(
                                                "• RMSSE (Root Mean Squared Scaled Error): Scaled by in-sample 1-step random walk MSE. "
                                                "Penalizes volatility without being biased by scale across departments.",
                                                className="small text-muted mb-1",
                                            ),
                                            html.P(
                                                "• WAPE (Weighted Absolute Percentage Error): sum(|y - y_hat|) / sum(y). "
                                                "Stable for intermittent retail demand where MAPE divides by zero.",
                                                className="small text-muted mb-1",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            className="shadow-sm border-0 mb-4",
                        ),
                        md=5,
                    ),
                ],
                className="g-3",
            )
        ],
        fluid=True,
    )


def build_app_layout(
    states: list[str],
    stores: list[str],
    categories: list[str],
    items: list[str],
) -> html.Div:
    """Assemble complete Dash web application layout."""
    return html.Div(
        [
            create_navbar(),
            dbc.Container(
                [
                    create_filter_bar(states, stores, categories, items),
                    dbc.Tabs(
                        [
                            dbc.Tab(create_forecasting_tab(), label="📈 Demand Forecasting", tab_id="tab-forecasting"),
                            dbc.Tab(create_reconciliation_tab(), label="🌳 Hierarchical Reconciliation (Darts)", tab_id="tab-reconciliation"),
                            dbc.Tab(create_inventory_tab(), label="⚙️ Inventory Optimization (OR-Tools)", tab_id="tab-inventory"),
                            dbc.Tab(create_benchmarks_tab(), label="🏆 Model Benchmarks (WAPE / RMSSE)", tab_id="tab-benchmarks"),
                        ],
                        id="tabs-main",
                        active_tab="tab-forecasting",
                        className="mb-4",
                    ),
                    dcc.Download(id="download-orders-csv"),
                ],
                fluid=True,
            ),
        ],
        style={"backgroundColor": "#f8fafc", "minHeight": "100vh"},
    )
