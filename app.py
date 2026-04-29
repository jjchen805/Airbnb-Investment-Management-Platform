"""
Airbnb Investment & Host Intelligence Platform
==============================================
Main app entry point.

Run with:
    python app.py

Prerequisites:
    pip install dash dash-bootstrap-components plotly pandas joblib catboost

Setup (first time only):
    1. Copy trained model files to models/{city}/
    2. python prepare_dashboard_data.py --city {city}
    3. python app.py
"""

import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx, no_update
import pandas as pd
import os

from layouts.home           import home_layout
from layouts.tab1_market    import market_layout
from layouts.tab2_advisor   import advisor_layout
from layouts.tab3_predictor import predictor_layout
from layouts.chat import chat_components

from callbacks.market_callbacks    import register_market_callbacks, ALL_DF
from callbacks.advisor_callbacks   import register_advisor_callbacks
from callbacks.predictor_callbacks import register_predictor_callbacks
from callbacks.chat_callbacks import register_chat_callbacks

# ── Load shared data ───────────────────────────────────────────────────────────
# ── Auto-discover cities from data/ folder ─────────────────────────────────────
CITIES = sorted([
    f.replace("dashboard_meta_", "").replace(".json", "")
    for f in os.listdir("data")
    if f.startswith("dashboard_meta_") and f.endswith(".json")
])

META = {city: json.load(open(f"data/dashboard_meta_{city}.json")) for city in CITIES}

# Derive human-readable labels from city code
# e.g. "sf" → "SF", "nyc" → "NYC", "chicago" → "Chicago"
def _city_label(code: str) -> str:
    overrides = {
        "sf":  "San Francisco",
        "nyc": "New York City",
        "la":  "Los Angeles",
        "dc":  "Washington D.C.",
    }
    return overrides.get(code, code.replace("_", " ").title())

CITY_LABELS = {city: _city_label(city) for city in CITIES}

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Airbnb Intelligence Platform",
)

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Navbar — hidden on home page, shown on tab pages
    html.Div(
        id="main-navbar",
        children=dbc.Navbar(
            dbc.Container([
                # Back to home
                dbc.Button(
                    [html.I(className="bi bi-arrow-left me-2"), "Home"],
                    id="btn-back-home",
                    color="outline-light",
                    size="sm",
                    className="me-3",
                    n_clicks=0,
                ),
                dbc.NavbarBrand(
                    "Airbnb Intelligence Platform",
                    className="fw-semibold",
                    style={"fontSize": "16px"},
                ),
                # City selector
                dcc.Dropdown(
                    id="city-select",
                    options=[{"label": CITY_LABELS[c], "value": c} for c in CITIES],
                    value="sf",
                    clearable=False,
                    style={"width": "180px", "color": "#000"},
                ),
                dbc.Badge("2025", color="secondary", className="ms-3"),
            ], fluid=True),
            color="dark", dark=True, className="mb-0 py-2",
        ),
        style={"display": "none"},   # hidden until user picks a tab
    ),

    # Tab bar — hidden on home page
    html.Div(
        id="main-tabbar",
        children=dbc.Container([
            dcc.Tabs(
                id="main-tabs",
                value="tab-home",
                className="mt-2",
                children=[
                    dcc.Tab(label="📊 Market Explorer",    value="tab-market"),
                    dcc.Tab(label="⭐ Superhost Advisor",  value="tab-advisor"),
                    dcc.Tab(label="💰 Investor Predictor", value="tab-predictor"),
                ],
            ),
        ], fluid=True, className="px-4"),
        style={"display": "none"},
    ),

    # Shared state
    dcc.Store(id="selected-city",      data="sf"),
    dcc.Store(id="selected-listing-id"),
    dcc.Store(id="adv-agent-context",  data={}),
    dcc.Store(id="inv-agent-context",  data=None),
    dcc.Store(id="inv-whatif-context", data=None),

    html.Div([
        dbc.Button(id="btn-persistent-market",    n_clicks=0),
        dbc.Button(id="btn-persistent-advisor",   n_clicks=0),
        dbc.Button(id="btn-persistent-predictor", n_clicks=0),
    ], style={"display": "none"}),

    # Page content
    html.Div(id="page-content", children=home_layout()),

    chat_components(),
])

app.clientside_callback(
    """
    function(m, a, p) {
        const triggered = window.dash_clientside.callback_context.triggered[0].prop_id.split('.')[0];
        return [
            triggered === 'btn-home-market'    ? m : window.dash_clientside.no_update,
            triggered === 'btn-home-advisor'   ? a : window.dash_clientside.no_update,
            triggered === 'btn-home-predictor' ? p : window.dash_clientside.no_update,
        ];
    }
    """,
    Output("btn-persistent-market",    "n_clicks"),
    Output("btn-persistent-advisor",   "n_clicks"),
    Output("btn-persistent-predictor", "n_clicks"),
    Input("btn-home-market",    "n_clicks"),
    Input("btn-home-advisor",   "n_clicks"),
    Input("btn-home-predictor", "n_clicks"),
    prevent_initial_call=True,
)


# ── Navigate from home → tab (landing page buttons) ───────────────────────────
@app.callback(
    Output("main-tabs",    "value"),
    Output("main-navbar",  "style"),
    Output("main-tabbar",  "style"),
    Output("page-content", "children"),
    Input("btn-persistent-market",    "n_clicks"),
    Input("btn-persistent-advisor",   "n_clicks"),
    Input("btn-persistent-predictor", "n_clicks"),
    Input("btn-back-home",            "n_clicks"),
    State("selected-city",            "data"),
    prevent_initial_call=True,
)
def handle_navigation(b_market, b_advisor, b_predictor, b_back, city):
    triggered = ctx.triggered_id
    city      = city or "sf"
    meta      = META.get(city, META[CITIES[0]])
    df        = ALL_DF.get(city, next(iter(ALL_DF.values())))
    visible   = {"display": "block"}
    hidden    = {"display": "none"}

    if triggered == "btn-back-home":
        return no_update, hidden, hidden, home_layout()

    if triggered == "btn-persistent-market":
        return "tab-market",    visible, visible, market_layout(meta)
    if triggered == "btn-persistent-advisor":
        return "tab-advisor",   visible, visible, advisor_layout(df)
    if triggered == "btn-persistent-predictor":
        return "tab-predictor", visible, visible, predictor_layout(meta)

    return "tab-market", hidden, hidden, home_layout()

# ── Sync city dropdown → store ────────────────────────────────────────────────
@app.callback(
    Output("selected-city", "data"),
    Input("city-select",    "value"),
)
def update_city(city):
    return city


# ── Render page content ───────────────────────────────────────────────────────
@app.callback(
    Output("page-content",  "children", allow_duplicate=True),
    Input("main-tabs",      "value"),
    Input("selected-city",  "data"),
    prevent_initial_call=True,
)
def render_tab(tab, city):
    if not tab or tab == "tab-home":
        return no_update          # ← don't overwrite home_layout()
    city = city or "sf"
    meta = META.get(city, META[CITIES[0]])
    df   = ALL_DF.get(city, next(iter(ALL_DF.values())))

    if tab == "tab-market":    return market_layout(meta)
    if tab == "tab-advisor":   return advisor_layout(df)
    if tab == "tab-predictor": return predictor_layout(meta)
    return no_update


# ── Register callbacks ────────────────────────────────────────────────────────
register_market_callbacks(app)
register_advisor_callbacks(app)
register_predictor_callbacks(app)
register_chat_callbacks(app)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
