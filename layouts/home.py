"""
Landing page — user type selector
"""
from dash import html
import dash_bootstrap_components as dbc


USER_TYPES = [
    {
        "btn_id":      "btn-home-market",
        "icon":        "bi-map",
        "title":       "Market Explorer",
        "subtitle":    "I want to explore the Airbnb market",
        "description": (
            "Browse thousands of listings on an interactive map. "
            "Filter by neighbourhood, room type, price, and Superhost status. "
            "See KPI summaries and detailed listing breakdowns."
        ),
        "color":       "primary",
        "btn_bg":      "#0071E3",
        "label":       "Explore the Market →",
    },
    {
        "btn_id":      "btn-home-advisor",
        "icon":        "bi-star-half",
        "title":       "Superhost Advisor",
        "subtitle":    "I already have a listing",
        "description": (
            "Select your existing listing and get a personalised Superhost "
            "probability score, SHAP-driven explanation, and ranked "
            "recommendations to improve your status."
        ),
        "color":       "warning",
        "btn_bg":      "#FF9F0A",
        "label":       "Advise My Listing →",
    },
    {
        "btn_id":      "btn-home-predictor",
        "icon":        "bi-graph-up-arrow",
        "title":       "Investor Predictor",
        "subtitle":    "I'm considering a new property",
        "description": (
            "Enter your planned listing details — neighbourhood, property type, "
            "amenities, host setup — and get an ML-based nightly price estimate "
            "with market benchmarks and key price drivers."
        ),
        "color":       "success",
        "btn_bg":      "#34C759",
        "label":       "Predict My Price →",
    },
]


def home_layout():
    cards = []
    for u in USER_TYPES:
        cards.append(
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            # Icon
                            html.Div(
                                html.I(className=f"bi {u['icon']}",
                                       style={"fontSize": "2rem", "color": u["btn_bg"]}),
                                style={"marginBottom": "20px"},
                            ),
                            # Title
                            html.H5(u["title"],
                                    style={"fontSize": "18px", "fontWeight": "600",
                                           "color": "#1D1D1F", "marginBottom": "4px"}),
                            # Subtitle
                            html.P(u["subtitle"],
                                   style={"fontSize": "13px", "color": "#6E6E73",
                                          "marginBottom": "16px"}),
                            # Description
                            html.P(u["description"],
                                   style={"fontSize": "14px", "color": "#3A3A3C",
                                          "lineHeight": "1.6", "minHeight": "72px",
                                          "marginBottom": "24px"}),
                            # Button
                            dbc.Button(
                                u["label"],
                                id=u["btn_id"],
                                n_clicks=0,
                                style={
                                    "background": u["btn_bg"],
                                    "border": "none",
                                    "borderRadius": "14px",
                                    "width": "100%",
                                    "fontSize": "14px",
                                    "fontWeight": "500",
                                    "padding": "11px 0",
                                    "color": "#fff",
                                },
                            ),
                        ], style={"padding": "32px"}),
                    ], className="user-type-card h-100"),
                ]),
                md=4,
            )
        )

    return html.Div([

        # ── Hero section ─────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.P("AIRBNB INTELLIGENCE PLATFORM",
                       style={"fontSize": "11px", "fontWeight": "600",
                              "letterSpacing": "2px", "color": "#6E6E73",
                              "marginBottom": "16px"}),
                html.H1("Smarter decisions,\nbetter returns.",
                        style={"fontSize": "clamp(2rem, 4vw, 3.2rem)",
                               "fontWeight": "700", "color": "#1D1D1F",
                               "letterSpacing": "-1px", "lineHeight": "1.1",
                               "marginBottom": "20px",
                               "whiteSpace": "pre-line"}),
                html.P(
                    "An end-to-end intelligence system for the short-term rental market. "
                    "Explore opportunities, optimise your listing, or predict returns on a new investment.",
                    style={"fontSize": "17px", "color": "#6E6E73",
                           "maxWidth": "520px", "margin": "0 auto 48px",
                           "lineHeight": "1.6"},
                ),
            ], style={"textAlign": "center"}),
        ], style={
            "padding": "72px 24px 56px",
            "background": "linear-gradient(180deg, #FFFFFF 0%, #F5F5F7 100%)",
        }),

        # ── Who are you section ───────────────────────────────────────────────
        html.Div([
            html.P("WHO ARE YOU?",
                   style={"fontSize": "11px", "fontWeight": "600",
                          "letterSpacing": "2px", "color": "#AEAEB2",
                          "textAlign": "center", "marginBottom": "32px"}),
            dbc.Row(cards, className="g-4 justify-content-center"),
        ], style={"padding": "0 24px 80px", "maxWidth": "1000px",
                  "margin": "0 auto"}),

    ])
