"""
Airbnb Investment & Host Intelligence Platform
==============================================
Main app entry point.

Run with:
    python app.py

Prerequisites:
    pip install dash dash-bootstrap-components plotly pandas joblib catboost

Setup (first time only):
    1. Copy trained model files to models/:
       price_model.pkl, price_features.pkl, price_meta.pkl
       superhost_model.pkl, superhost_features.pkl, superhost_meta.pkl
    2. python prepare_dashboard_data.py
    3. python app.py
"""

import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd

from layouts.tab1_market    import market_layout
from layouts.tab2_advisor   import advisor_layout
from layouts.tab3_predictor import predictor_layout

from callbacks.market_callbacks    import register_market_callbacks, ALL_DF
from callbacks.advisor_callbacks   import register_advisor_callbacks
from callbacks.predictor_callbacks import register_predictor_callbacks

# ── Load meta for tab layouts (neighbourhoods, property types) ────────────────
CITIES = ["sf", "nyc", "chicago"]
META = {city: json.load(open(f"data/dashboard_meta_{city}.json")) for city in CITIES}

CITY_LABELS = {"sf": "San Francisco", "nyc": "New York City", "chicago": "Chicago"}

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Airbnb Intelligence Platform",
)

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.Span("🏠", style={"marginRight": "8px"}),
                "Airbnb Investment & Host Intelligence Platform",
            ], className="fw-semibold fs-5"),

            # City selector in navbar
            dcc.Dropdown(
                id="city-select",
                options=[{"label": CITY_LABELS[c], "value": c} for c in CITIES],
                value="sf",
                clearable=False,
                style={"width": "200px", "color": "#000"},
            ),

            dbc.Badge("2025", color="secondary", className="ms-3"),
        ], fluid=True),
        color="dark", dark=True, className="mb-0 py-2",
    ),

    dbc.Container([
        html.P(
            "Explore market opportunities, evaluate listing performance, "
            "and receive ML-based pricing and Superhost recommendations.",
            className="text-muted mt-2 mb-0 small",
        )
    ], fluid=True, className="px-4"),

    dbc.Container([
        dcc.Tabs(
            id="main-tabs",
            value="tab-market",
            className="mt-3",
            children=[
                dcc.Tab(label="📊 Market Explorer",    value="tab-market"),
                dcc.Tab(label="⭐ Superhost Advisor",  value="tab-advisor"),
                dcc.Tab(label="💰 Investor Predictor", value="tab-predictor"),
            ],
        ),

        # Shared state across tabs
        dcc.Store(id="selected-city",      data="sf"),
        dcc.Store(id="selected-listing-id"),

        html.Div(id="tab-content", className="mt-3"),

    ], fluid=True, className="px-4"),
])


# ── Callback: sync city dropdown → dcc.Store ─────────────────────────────────
@app.callback(
    Output("selected-city", "data"),
    Input("city-select",    "value"),
)
def update_selected_city(city):
    return city


# ── Callback: tab routing — passes city-specific meta to each layout ──────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs",    "value"),
    Input("selected-city", "data"),
)
def render_tab(tab, city):
    city = city or "sf"
    meta = META.get(city, META[CITIES[0]])
    df   = ALL_DF.get(city, next(iter(ALL_DF.values())))

    if tab == "tab-market":
        return market_layout(meta)
    elif tab == "tab-advisor":
        return advisor_layout(df)
    elif tab == "tab-predictor":
        return predictor_layout(meta)
    return html.Div()


# ── Register callbacks ────────────────────────────────────────────────────────
register_market_callbacks(app)
register_advisor_callbacks(app)
register_predictor_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True, port=8050)