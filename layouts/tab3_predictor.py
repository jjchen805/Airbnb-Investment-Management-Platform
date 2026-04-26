"""
Tab 3 — Investor Price Predictor Layout
========================================
Output panel driven by local SHAP values + amenity gap analysis.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

NEIGHBOURHOODS = [
    "Bernal Heights", "Castro/Upper Market", "Chinatown",
    "Downtown/Civic Center", "Excelsior", "Financial District",
    "Haight Ashbury", "Inner Richmond", "Inner Sunset", "Marina",
    "Mission", "Nob Hill", "Noe Valley", "North Beach", "Other",
    "Outer Richmond", "Outer Sunset", "Pacific Heights",
    "Russian Hill", "South of Market", "Western Addition",
]

PROPERTY_TYPES = [
    "Entire condo", "Entire guest suite", "Entire home",
    "Entire rental unit", "Entire serviced apartment",
    "Private room in condo", "Private room in home",
    "Private room in rental unit", "Room in boutique hotel",
    "Room in hotel", "Other",
]

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

# Apple system colors
C = {
    "blue":   "#0071E3",
    "green":  "#34C759",
    "orange": "#FF9F0A",
    "red":    "#FF3B30",
    "gray1":  "#1D1D1F",
    "gray2":  "#3A3A3C",
    "gray3":  "#6E6E73",
    "gray4":  "#AEAEB2",
    "gray5":  "#C7C7CC",
    "gray6":  "#F2F2F7",
}

def predictor_layout(meta: dict):
    neighbourhoods = sorted([r["neighbourhood_top"] for r in meta["neighbourhoods"]])
    property_types = sorted([r["property_type_simple"] for r in meta["property_types"]])
    return html.Div([
        html.H4("Airbnb Price Predictor", 
                style={"fontSize": "22px", "fontWeight": "700",
                           "color": C["gray1"], "marginBottom": "4px"}),
        html.P(
            "Enter your planned listing details and get an ML-based nightly price estimate "
            "benchmarked against the market.",
            style={"fontSize": "14px", "color": C["gray3"], "marginBottom": "24px"},
        ),

        dbc.Row([
            # ── LEFT: Input form ─────────────────────────────────────────────
            dbc.Col([

                dbc.Card([
                    dbc.CardHeader("LOCATION"),
                    dbc.CardBody([
                        dbc.Label("Neighborhood"),
                        dcc.Dropdown(
                            id="inv-neighbourhood",
                            options=[{"label": n, "value": n} for n in neighbourhoods],
                            value="Mission", clearable=False,
                        ),
                    ]),
                ], className="mb-3"),

                dbc.Card([
                    dbc.CardHeader("PROPERTY SETUP"),
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
                ], className="mb-3"),

                dbc.Card([
                    dbc.CardHeader("HOST SETUP"),
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
                ], className="mb-3"),

                dbc.Card([
                    dbc.CardHeader("AMENITIES"),
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
                ], className="mb-3"),

                dbc.Button(
                    "🔮  Predict Nightly Price",
                    id="inv-predict-btn",
                    size="lg",
                    className="w-100 mb-4",
                    style={"background": "#0071E3", "border": "none", "borderRadius": "14px", "fontSize": "15px", "fontWeight": "600"},
                    n_clicks=0,
                ),

            ], md=5),

            # ── RIGHT: Output panel ──────────────────────────────────────────
            dbc.Col([
                html.Div(id="inv-output-panel", children=[_empty_output_panel()])
            ], md=7),
        ]),
    ], style={"padding": "24px 32px"})


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
    ], )


# ── Output panel ──────────────────────────────────────────────────────────────

def build_output_panel(result: dict,whatif: dict = None):
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

    panels = [
        _predicted_price_card(predicted_price),
        _market_comparison_card(predicted_price, neighbourhood, nbhd_median,
                                pct_vs_nbhd, property_type, prop_median, pct_vs_prop),
        _shap_drivers_card(drivers, predicted_price),
        _amenity_tips_card(amenity_gaps, neighbourhood, pct_vs_nbhd, result.get("city", "sf")),
    ]

    # Show what-if results when available
    if whatif and whatif.get("scenarios"):
        panels.append(_whatif_card(whatif))

    panels.append(_investor_agent_panel())
    return html.Div(panels, style={"display": "flex", "flexDirection": "column", "gap": "16px"})


def _predicted_price_card(predicted_price):
    low  = predicted_price * 0.88
    high = predicted_price * 1.12
    return dbc.Card([
        dbc.CardBody([
            html.P("Predicted nightly price", className="text-muted mb-1 small"),
            html.H1(
                f"${predicted_price:,.0f}",
                className="display-4 fw-bold mb-0", 
                style={"color": "#0071E3", "letterSpacing": "-2px"},
            ),
            html.P(
                f"Estimated range: ${low:,.0f} – ${high:,.0f}",
                className="text-muted small mt-1"
            ),
        ], className="text-center py-4")
    ])


def _market_comparison_card(predicted_price, neighbourhood, nbhd_median, pct_vs_nbhd, property_type, prop_median, pct_vs_prop):
    def badge(pct):
        color = "success" if pct >= 0 else "danger"
        sign  = "+" if pct >= 0 else ""
        return dbc.Badge(f"{sign}{pct:.1f}% vs avg", color=color, className="ms-2 fs-6")

    return dbc.Card([
        dbc.CardHeader("MARKET COMPARISON"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Neighborhood median", className="text-muted small mb-0"),
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
    ])


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
        sign       = "+" if d["direction"] == "up" else "-"
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
                            "height": "8px",
                            "background": color,
                            "borderRadius": "4px",
                            "transition": "width 0.4s ease",
                        }),
                        style={"background": "#F2F2F7", "borderRadius": "4px"}
                    ),
                    width=5
                ),
                dbc.Col(
                    html.Span(dollar_str, style={"fontSize": "12px", "color": color, "fontWeight": "600"}),
                    width=2
                ),
            ], align="center", className="mb-2")
        )

    return dbc.Card([
        dbc.CardHeader([
            "KEY PRICE DRIVERS",
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
    ])


def _amenity_tips_card(amenity_gaps: list, neighbourhood: str, pct_vs_nbhd: float, city: str = "sf"):
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
            "Your predicted price is below the neighborhood average. "
            "Adding premium amenities (gym, pool, hot tub) could close this gap significantly."
        )
    elif pct_vs_nbhd > 20:
        positioning_tip = (
            "Your price is well above the neighborhood average. "
            "Prioritize earning strong early reviews to sustain occupancy at this price point."
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
                f"These two settings increase booking conversion and are associated with "
                f"a higher median price in {city.upper()}.",
                className="text-muted"
            ),
        ], className="border-0 px-0 py-2")
    )

    return dbc.Card([
        dbc.CardHeader("INVESTOR TIPS"),
        dbc.CardBody([
            html.P(
                f"Based on your listing configuration and {city.upper()} market data:",
                className="text-muted small mb-2"
            ),
            dbc.ListGroup(tip_items, flush=True),
        ])
    ], )

def _whatif_card(whatif: dict):
    base_price = whatif.get("base_price", 0)
    budget     = whatif.get("budget", 0)
    scenarios  = whatif.get("scenarios", [])

    rows = []
    for i, s in enumerate(scenarios):
        uplift = s.get("uplift_usd", 0)
        color  = "#34C759" if uplift > 0 else "#FF3B30"
        rows.append(
            html.Div([
                html.Div([
                    html.Span(f"#{i+1}  {s['name']}",
                              style={"fontSize": "13px", "fontWeight": "600",
                                     "color": "#1D1D1F"}),
                    html.Span(f"  +${uplift:,.0f}/night",
                              style={"fontSize": "13px", "fontWeight": "600",
                                     "color": color}),
                ], style={"marginBottom": "2px"}),
                html.P(
                    f"Est. cost: ${s.get('estimated_budget', 0):,.0f}  •  "
                    + ",  ".join(s.get("changes", [])),
                    style={"fontSize": "12px", "color": "#6E6E73", "margin": "0 0 12px"}
                ),
            ])
        )

        narrative = whatif.get("llm_narrative")
        narrative_section = []
        if narrative:
            narrative_section = [
                html.Hr(style={"margin": "12px 0", "borderColor": "#F2F2F7"}),
                dcc.Markdown(narrative,
                            style={"fontSize": "13px", "color": "#3A3A3C",
                                    "lineHeight": "1.6"},
                            className="markdown-body"),
            ]

    return dbc.Card([
        dbc.CardBody([
            html.P("WHAT-IF SCENARIOS", style={
                "fontSize": "11px", "fontWeight": "600",
                "color": "#AEAEB2", "letterSpacing": "0.8px", "marginBottom": "4px",
            }),
            html.P(
                f"Top upgrades within ${budget:,.0f} budget, ranked by price uplift "
                f"and booking competitiveness.",
                style={"fontSize": "13px", "color": "#6E6E73", "marginBottom": "16px"},
            ),
            *rows,
            *narrative_section,
        ]),
    ])

def _investor_agent_panel():
    """AI brief, Q&A, and what-if scenario panel for the Investor Predictor."""
    from dash import html, dcc
    import dash_bootstrap_components as dbc
 
    C = {
        "blue":  "#0071E3",
        "gray1": "#1D1D1F",
        "gray3": "#6E6E73",
        "gray4": "#AEAEB2",
        "gray6": "#F2F2F7",
    }
 
    return dbc.Card([
        dbc.CardBody([
            html.P("AI INVESTMENT ADVISOR", style={
                "fontSize": "11px", "fontWeight": "600",
                "color": C["gray4"], "letterSpacing": "0.8px",
                "marginBottom": "4px",
            }),
            html.P(
                "Generate an investment brief, run what-if scenarios within a budget, "
                "or ask a question about this prediction.",
                style={"fontSize": "13px", "color": C["gray3"], "marginBottom": "16px"},
            ),
 
            # Budget input
            dbc.Row([
                dbc.Col([
                    html.Label("Budget ($)", style={"fontSize": "12px", "color": C["gray3"]}),
                    dbc.Input(
                        id="inv-whatif-budget",
                        type="number", value=1500, min=0, step=100,
                        style={"borderRadius": "10px", "fontSize": "13px",
                               "border": "1px solid #D1D1D6"},
                    ),
                ], width=4),
            ]),
 
            # Question input
            dbc.Textarea(
                id="inv-agent-question",
                placeholder="e.g. Should I prioritize adding parking or a gym?",
                style={
                    "fontSize": "13px", "borderRadius": "10px",
                    "border": "1px solid #D1D1D6",
                    "padding": "10px 12px", "resize": "none",
                    "height": "72px", "marginBottom": "10px",
                    "width": "100%", "marginTop": "8px",
                },
            ),
 
            # Buttons
            dbc.Row([
                dbc.Col(
                    dbc.Button("Investment brief",
                               id="inv-agent-brief-btn", n_clicks=0,
                               style={"background": C["blue"], "border": "none",
                                      "borderRadius": "10px", "fontSize": "13px",
                                      "fontWeight": "500", "width": "100%"}),
                    width=4
                ),
                dbc.Col(
                    dbc.Button("What-if scenarios",
                               id="inv-whatif-generate-btn", n_clicks=0,
                               style={"background": "#F2F2F7", "border": "none",
                                      "borderRadius": "10px", "fontSize": "13px",
                                      "fontWeight": "500",
                                      "width": "100%"}),
                    width=4
                ),
                dbc.Col(
                    dbc.Button("Ask",
                               id="inv-agent-ask-btn", n_clicks=0,
                               style={"background": "#F2F2F7", "border": "none",
                                      "borderRadius": "10px", "fontSize": "13px",
                                      "fontWeight": "500",
                                      "width": "100%"}),
                    width=4
                ),
            ], className="mb-3 g-2"),
 
            # Response area
            dcc.Loading(
                dcc.Markdown(id="inv-agent-response",
                             style={"fontSize": "13px", "color": C["gray1"],
                                    "lineHeight": "1.6"},
                             className="markdown-body"),
                type="circle", color=C["blue"],
            ),
        ]),
    ], className="mt-3")