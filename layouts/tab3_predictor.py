"""
Tab 3 — Investor Price Predictor Layout
========================================
Output panel driven by local SHAP values + amenity gap analysis.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

AMENITIES = [
    ("has_wifi",            "WiFi"),
    ("has_kitchen",         "Kitchen"),
    ("has_washer",          "Washer"),
    ("has_dryer",           "Dryer"),
    ("has_parking",         "Parking"),
    ("has_air_conditioning","AC"),
    ("has_heating",         "Heating"),
    ("has_tv",              "TV"),
    ("has_self_check-in",   "Self check-in"),
    ("has_coffee",          "Coffee maker"),
    ("has_hair_dryer",      "Hair dryer"),
    ("has_iron",            "Iron"),
    ("has_gym",             "Gym"),
    ("has_pool",            "Pool"),
    ("has_hot_tub",         "Hot tub"),
    ("has_elevator",        "Elevator"),
]

DEFAULT_ON = {"has_wifi", "has_kitchen", "has_heating", "has_tv", "has_hair_dryer"}


def predictor_layout(meta: dict):
    neighbourhoods = sorted([r["neighbourhood_top"] for r in meta["neighbourhoods"]])
    property_types = sorted([r["property_type_simple"] for r in meta["property_types"]])
    return dbc.Container([
        html.H4("Investor Price Predictor", className="mt-3 mb-1 fw-semibold"),
        html.P(
            "Enter your planned listing details and get an ML-based nightly price estimate "
            "benchmarked against the SF market.",
            className="text-muted mb-4"
        ),

        dbc.Row([
            # ── LEFT: Input form ─────────────────────────────────────────────
            dbc.Col([

                dbc.Card([
                    dbc.CardHeader("📍 Location"),
                    dbc.CardBody([
                        dbc.Label("Neighbourhood"),
                        dcc.Dropdown(
                            id="inv-neighbourhood",
                            options=[{"label": n, "value": n} for n in neighbourhoods],
                            value="Mission", clearable=False,
                        ),
                    ]),
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader("🏠 Property Setup"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Property Type"),
                                dcc.Dropdown(
                                    id="inv-property-type",
                                    options=[{"label": p, "value": p} for p in property_types],
                                    value="Entire rental unit", clearable=False,
                                ),
                            ], width=12, className="mb-3"),
                        ]),
                        dbc.Row([
                            dbc.Col([dbc.Label("Accommodates"), dbc.Input(id="inv-accommodates", type="number", value=4, min=1, max=16)], width=4),
                            dbc.Col([dbc.Label("Bedrooms"),     dbc.Input(id="inv-bedrooms",     type="number", value=1, min=0, max=10)], width=4),
                            dbc.Col([dbc.Label("Bathrooms"),    dbc.Input(id="inv-bathrooms",    type="number", value=1, min=0.5, max=8, step=0.5)], width=4),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([dbc.Label("Beds"),           dbc.Input(id="inv-beds",           type="number", value=1, min=1, max=16)], width=4),
                            dbc.Col([dbc.Label("Min nights"),     dbc.Input(id="inv-min-nights",     type="number", value=2, min=1, max=365)], width=4),
                            dbc.Col([dbc.Label("Avail. / yr"),    dbc.Input(id="inv-availability-365", type="number", value=200, min=0, max=365)], width=4),
                        ]),
                    ]),
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader("👤 Host Setup"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Instant bookable?"),
                                dbc.RadioItems(
                                    id="inv-instant-bookable",
                                    options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                                    value=0, inline=True,
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Superhost?"),
                                dbc.RadioItems(
                                    id="inv-superhost",
                                    options=[{"label": "Yes", "value": 1.0}, {"label": "No", "value": 0.0}],
                                    value=0.0, inline=True,
                                ),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Total host listings"),
                                dbc.Input(id="inv-host-listings", type="number", value=1, min=1, max=100),
                            ], width=6),
                        ]),
                    ]),
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader("✨ Amenities"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(
                                dbc.Checklist(
                                    options=[{"label": label, "value": 1}],
                                    value=[1] if col in DEFAULT_ON else [],
                                    id=f"inv-{col}",
                                    switch=True,
                                ),
                                width=6, className="mb-1"
                            )
                            for col, label in AMENITIES
                        ]),
                        html.Hr(className="my-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Total amenity count (approx.)"),
                                dbc.Input(id="inv-amenity-count", type="number", value=25, min=0, max=100),
                            ], width=6),
                        ]),
                    ]),
                ], className="mb-3 shadow-sm"),

                dbc.Button(
                    "🔮  Predict Nightly Price",
                    id="inv-predict-btn",
                    color="primary", size="lg",
                    className="w-100 mb-4",
                    n_clicks=0,
                ),

            ], md=5),

            # ── RIGHT: Output panel ──────────────────────────────────────────
            dbc.Col([
                html.Div(id="inv-output-panel", children=[_empty_output_panel()])
            ], md=7),
        ]),
    ], fluid=True)


def _empty_output_panel():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div("🏡", style={"fontSize": "48px", "textAlign": "center"}),
                html.P(
                    "Fill in your listing details and click Predict Nightly Price.",
                    className="text-muted text-center mt-3"
                ),
            ], className="py-5")
        ])
    ], className="shadow-sm")


# ── Output panel ──────────────────────────────────────────────────────────────

def build_output_panel(result: dict):
    """
    Build the full result panel from the predict_price() result dict.
    """
    predicted_price = result["predicted_price"]
    nbhd_median     = result["nbhd_median"]
    prop_median     = result["prop_median"]
    pct_vs_nbhd     = result["pct_vs_nbhd"]
    neighbourhood   = result["neighbourhood"]
    property_type   = result["property_type"]
    drivers         = result["drivers"]
    amenity_gaps    = result["amenity_gaps"]

    pct_vs_prop = ((predicted_price - prop_median) / prop_median) * 100

    return html.Div([
        _predicted_price_card(predicted_price),
        _market_comparison_card(predicted_price, neighbourhood, nbhd_median, pct_vs_nbhd, property_type, prop_median, pct_vs_prop),
        _shap_drivers_card(drivers, predicted_price),
        _amenity_tips_card(amenity_gaps, neighbourhood, pct_vs_nbhd),
    ])


def _predicted_price_card(predicted_price):
    low  = predicted_price * 0.88
    high = predicted_price * 1.12
    return dbc.Card([
        dbc.CardBody([
            html.P("Predicted nightly price", className="text-muted mb-1 small"),
            html.H1(
                f"${predicted_price:,.0f}",
                className="display-4 fw-bold text-primary mb-0",
                style={"letterSpacing": "-1px"},
            ),
            html.P(
                f"Estimated range: ${low:,.0f} – ${high:,.0f}",
                className="text-muted small mt-1"
            ),
        ], className="text-center py-4")
    ], className="shadow mb-3 border-primary border-2")


def _market_comparison_card(predicted_price, neighbourhood, nbhd_median, pct_vs_nbhd, property_type, prop_median, pct_vs_prop):
    def badge(pct):
        color = "success" if pct >= 0 else "danger"
        sign  = "+" if pct >= 0 else ""
        return dbc.Badge(f"{sign}{pct:.1f}% vs avg", color=color, className="ms-2 fs-6")

    return dbc.Card([
        dbc.CardHeader("📊 Market Comparison"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Neighbourhood median", className="text-muted small mb-0"),
                    html.H5([f"${nbhd_median:,.0f}", badge(pct_vs_nbhd)], className="mb-0"),
                    html.Small(neighbourhood, className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.P("Property type median", className="text-muted small mb-0"),
                    html.H5([f"${prop_median:,.0f}", badge(pct_vs_prop)], className="mb-0"),
                    html.Small(property_type, className="text-muted"),
                ], width=6),
            ]),
        ])
    ], className="shadow-sm mb-3")


def _shap_drivers_card(drivers: list, predicted_price: float):
    """
    Waterfall-style SHAP driver chart.
    Each bar shows direction (green=positive, red=negative) and dollar impact.
    """
    max_abs = max(abs(d["dollar_impact"]) for d in drivers) if drivers else 1

    rows = []
    for d in drivers:
        pct_width  = int(min(abs(d["dollar_impact"]) / max_abs * 100, 100))
        color      = "#198754" if d["direction"] == "up" else "#dc3545"
        sign       = "+" if d["direction"] == "up" else ""
        dollar_str = f"{sign}${abs(d['dollar_impact']):,.0f}"

        rows.append(
            dbc.Row([
                dbc.Col(
                    html.Span(d["label"], className="text-muted", style={"fontSize": "13px"}),
                    width=5
                ),
                dbc.Col(
                    html.Div(
                        html.Div(style={
                            "width": f"{pct_width}%",
                            "height": "10px",
                            "background": color,
                            "borderRadius": "4px",
                            "transition": "width 0.4s ease",
                        }),
                        style={"background": "#e9ecef", "borderRadius": "4px"}
                    ),
                    width=5
                ),
                dbc.Col(
                    html.Span(dollar_str, style={"fontSize": "12px", "color": color, "fontWeight": "500"}),
                    width=2
                ),
            ], align="center", className="mb-2")
        )

    return dbc.Card([
        dbc.CardHeader([
            "🔑 Key Price Drivers",
            html.Span(
                " — how each factor pushed your price up or down from the SF baseline",
                className="text-muted fw-normal small"
            ),
        ]),
        dbc.CardBody([
            html.P(
                f"SF baseline (avg prediction): ${drivers[0]['dollar_impact'] and '—' or '—'}",
                className="text-muted small mb-3"
            ) if not drivers else None,
            *rows,
            html.Hr(className="my-2"),
            html.P(
                "Dollar values show each feature's approximate contribution to your predicted price, "
                "computed from CatBoost TreeSHAP values.",
                className="text-muted small mb-0"
            ),
        ])
    ], className="shadow-sm mb-3")


def _amenity_tips_card(amenity_gaps: list, neighbourhood: str, pct_vs_nbhd: float):
    """
    Amenity-aware investor tips.
    Each unchecked amenity with meaningful market lift gets a targeted tip,
    grounded in actual SF data for this neighbourhood.
    """
    tip_items = []

    # Amenity-specific tips from data
    for gap in amenity_gaps:
        label       = gap["label"]
        market_lift = gap["market_lift"]
        tip_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Span(f"Add {label}", className="fw-semibold"),
                    dbc.Badge(
                        f"+${market_lift:,.0f} median lift in {neighbourhood}",
                        color="success", className="ms-2 small"
                    ),
                ]),
                html.Small(
                    f"Listings in {neighbourhood} with {label.lower()} have a "
                    f"${market_lift:,.0f} higher median nightly price than those without.",
                    className="text-muted"
                ),
            ], className="border-0 px-0 py-2")
        )

    # Positioning tip (based on SHAP/market output, not hardcoded)
    if pct_vs_nbhd < -15:
        positioning_tip = (
            "Your predicted price is below the neighbourhood average. "
            "Adding premium amenities (gym, pool, hot tub) could close this gap significantly."
        )
    elif pct_vs_nbhd > 20:
        positioning_tip = (
            "Your price is well above the neighbourhood average. "
            "Prioritise earning strong early reviews to sustain occupancy at this price point."
        )
    else:
        positioning_tip = (
            "Your pricing is in line with the market. "
            "Gaining Superhost status early tends to improve both occupancy and price ceiling."
        )

    tip_items.append(
        dbc.ListGroupItem([
            html.Span("Market positioning", className="fw-semibold"),
            html.Br(),
            html.Small(positioning_tip, className="text-muted"),
        ], className="border-0 px-0 py-2")
    )

    # Generic but always-useful tip
    tip_items.append(
        dbc.ListGroupItem([
            html.Span("Enable self check-in & instant book", className="fw-semibold"),
            html.Br(),
            html.Small(
                "These two settings increase booking conversion and are associated with "
                "a $33–$35 higher median price in SF.",
                className="text-muted"
            ),
        ], className="border-0 px-0 py-2")
    )

    return dbc.Card([
        dbc.CardHeader("💡 Investor Tips"),
        dbc.CardBody([
            html.P(
                "Based on your listing configuration and SF market data:",
                className="text-muted small mb-2"
            ),
            dbc.ListGroup(tip_items, flush=True),
        ])
    ], className="shadow-sm")
