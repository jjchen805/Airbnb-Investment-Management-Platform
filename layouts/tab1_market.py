"""
Tab 1 — Market Explorer Layout
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def market_layout(meta: dict):
    neighbourhoods = sorted([r["neighbourhood_top"] for r in meta["neighbourhoods"]])
    property_types = sorted([r["property_type_simple"] for r in meta["property_types"]])
    room_types     = ["Entire home/apt", "Private room", "Hotel room", "Shared room"]

    return html.Div([

        # ── KPI Cards ─────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(_kpi_card("inv-kpi-listings",  "Listings shown",      "—", "bi-house-door",      "#0071E3"), md=3),
            dbc.Col(_kpi_card("inv-kpi-price",     "Avg nightly price",   "—", "bi-currency-dollar", "#FF9F0A"), md=3),
            dbc.Col(_kpi_card("inv-kpi-superhost", "Superhost %",         "—", "bi-star",            "#FF375F"), md=3),
            dbc.Col(_kpi_card("inv-kpi-sentiment", "Avg sentiment",       "—", "bi-chat-heart",      "#34C759"), md=3),
        ], className="mb-4 g-3"),

        # ── Main row ───────────────────────────────────────────────────────────
        dbc.Row([

            # Left: Filters
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([

                        html.P("Neighbourhood", className="filter-section-label"),
                        dcc.Dropdown(
                            id="mkt-neighbourhood",
                            options=[{"label": "All", "value": "All"}] +
                                    [{"label": n, "value": n} for n in neighbourhoods],
                            value="All", clearable=False, className="mb-3",
                        ),

                        html.P("Room type", className="filter-section-label"),
                        dcc.Dropdown(
                            id="mkt-room-type",
                            options=[{"label": "All", "value": "All"}] +
                                    [{"label": r, "value": r} for r in room_types],
                            value="All", clearable=False, className="mb-3",
                        ),

                        html.P("Property type", className="filter-section-label"),
                        dcc.Dropdown(
                            id="mkt-property-type",
                            options=[{"label": "All", "value": "All"}] +
                                    [{"label": p, "value": p} for p in property_types],
                            value="All", clearable=False, className="mb-3",
                        ),

                        html.P("Price range ($/night)", className="filter-section-label"),
                        dcc.RangeSlider(
                            id="mkt-price-range",
                            min=10, max=3000, step=10,
                            value=[10, 3000],
                            marks={10: "$10", 500: "$500", 1000: "$1k", 3000: "$3k"},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-3",
                        ),

                        html.P("Superhost", className="filter-section-label"),
                        dbc.RadioItems(
                            id="mkt-superhost",
                            options=[
                                {"label": "All",           "value": "all"},
                                {"label": "Superhost",     "value": "yes"},
                                {"label": "Non-superhost", "value": "no"},
                            ],
                            value="all", inline=False,
                            className="mb-4",
                            style={"fontSize": "13px"},
                        ),

                        dbc.Button(
                            [html.I(className="bi bi-arrow-counterclockwise me-2"),
                             "Reset filters"],
                            id="mkt-reset-btn",
                            color="outline-secondary",
                            size="sm",
                            className="w-100",
                        ),
                    ]),
                ], className="h-100"),
            ], md=2),

            # Center: Map
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id="mkt-map",
                            config={"scrollZoom": True, "displayModeBar": False},
                            style={"height": "620px"},
                        ),
                    ], style={"padding": "6px !important"}),
                ]),
            ], md=7),

            # Right: Detail card
            dbc.Col([
                html.Div(id="mkt-detail-panel", children=[_empty_detail_card()])
            ], md=3),

        ], className="g-3"),
    ], style={"padding": "24px"})


# ── Sub-components ─────────────────────────────────────────────────────────────

def _kpi_card(card_id, label, value, icon, accent):
    return html.Div([
        dbc.Row([
            dbc.Col(
                html.Div(
                    html.I(className=f"bi {icon}",
                           style={"fontSize": "20px", "color": accent}),
                    style={
                        "width": "40px", "height": "40px",
                        "borderRadius": "10px",
                        "background": f"{accent}14",
                        "display": "flex", "alignItems": "center",
                        "justifyContent": "center",
                    }
                ),
                width="auto"
            ),
            dbc.Col([
                html.P(label, className="kpi-label mb-0"),
                html.Div(value, id=card_id, className="kpi-value"),
            ]),
        ], align="center", className="g-2"),
    ], className="kpi-card")


def _empty_detail_card():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(
                    html.I(className="bi bi-cursor",
                           style={"fontSize": "28px", "color": "#C7C7CC"}),
                    style={"marginBottom": "12px"},
                ),
                html.P("Click a listing on the map",
                       style={"fontSize": "14px", "color": "#AEAEB2",
                              "textAlign": "center", "margin": 0}),
            ], style={
                "display": "flex", "flexDirection": "column",
                "alignItems": "center", "justifyContent": "center",
                "minHeight": "300px",
            }),
        ]),
    ])


def build_detail_card(row: dict):
    superhost     = row.get("host_is_superhost", 0)
    price         = row.get("price_clean")
    price_str     = f"${price:,.0f} / night" if price and price == price else "N/A"
    sentiment     = row.get("sentiment_polarity_mean")
    sent_str      = f"{sentiment:.2f}" if sentiment == sentiment and sentiment is not None else "N/A"
    reviews       = int(row.get("number_of_reviews", 0))
    amenities     = int(row.get("amenity_count", 0))

    # AFTER
    themes = {
        "Cleanliness":   (row.get("theme_cleanliness_mean"),   row.get("theme_cleanliness_positive_mean"),   row.get("theme_cleanliness_negative_mean")),
        "Communication": (row.get("theme_communication_mean"), row.get("theme_communication_positive_mean"), row.get("theme_communication_negative_mean")),
        "Check-in":      (row.get("theme_checkin_mean"),       row.get("theme_checkin_positive_mean"),       row.get("theme_checkin_negative_mean")),
        "Location":      (row.get("theme_location_mean"),      row.get("theme_location_positive_mean"),      row.get("theme_location_negative_mean")),
        "Amenities":     (row.get("theme_amenities_mean"),     row.get("theme_amenities_positive_mean"),     row.get("theme_amenities_negative_mean")),
        "Value":         (row.get("theme_value_mean"),         row.get("theme_value_positive_mean"),         row.get("theme_value_negative_mean")),
        "Comfort":       (row.get("theme_comfort_mean"),       row.get("theme_comfort_positive_mean"),       row.get("theme_comfort_negative_mean")),
        "Accuracy":      (row.get("theme_accuracy_mean"),      row.get("theme_accuracy_positive_mean"),      row.get("theme_accuracy_negative_mean")),
    }

    superhost_pill = html.Span(
        "⭐ Superhost" if superhost else "Non-superhost",
        style={
            "fontSize": "11px", "fontWeight": "600",
            "padding": "3px 10px", "borderRadius": "20px",
            "background": "#FFF3CD" if superhost else "#F2F2F7",
            "color": "#92600A" if superhost else "#6E6E73",
        }
    )

    return dbc.Card([
        dbc.CardBody([
            # Name + badge
            html.P(row.get("name", "Listing")[:48],
                   style={"fontSize": "14px", "fontWeight": "600",
                          "color": "#1D1D1F", "marginBottom": "8px",
                          "lineHeight": "1.3"}),
            html.Div(superhost_pill, style={"marginBottom": "16px"}),

            # Info rows
            *[_detail_row(icon, text) for icon, text in [
                ("bi-geo-alt",        row.get("neighbourhood_cleansed", "—")),
                ("bi-house",          row.get("room_type", "—")),
                ("bi-currency-dollar",price_str),
                ("bi-chat-left-text", f"{reviews} reviews"),
                ("bi-emoji-smile",    f"Sentiment {sent_str}"),
                ("bi-check2-all",     f"{amenities} amenities"),
            ]],

            html.Hr(),

            # Theme bars
            html.P("Review themes",
                   style={"fontSize": "11px", "fontWeight": "600",
                          "color": "#AEAEB2", "textTransform": "uppercase",
                          "letterSpacing": "0.5px", "marginBottom": "10px"}),
            *[_theme_bar(label, score, pos, neg) for label, (score, pos, neg) in themes.items()
              if score is not None and score == score],

            html.Div(
                html.A("View on Airbnb →",
                       href=row.get("listing_url", "#"),
                       target="_blank",
                       style={"fontSize": "13px", "color": "#0071E3",
                              "textDecoration": "none", "fontWeight": "500"}),
                style={"marginTop": "12px"}
            ),
        ]),
    ])


def _detail_row(icon, text):
    return html.Div([
        html.I(className=f"bi {icon}",
               style={"fontSize": "13px", "color": "#AEAEB2",
                      "width": "18px", "marginRight": "8px"}),
        html.Span(str(text), style={"fontSize": "13px", "color": "#3A3A3C"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"})


def _theme_bar(label: str, score: float, pos: float = 0, neg: float = 0):
    pct   = min(int(score * 100), 100)
    pos = pos or 0
    neg = neg or 0
    net = pos - neg
    color = "#34C759" if net > 0.20 else "#FF9F0A" if net > 0.05 else "#FF3B30"
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "12px", "color": "#6E6E73", "width": "110px"}),
            html.Div(
                html.Div(style={
                    "width": f"{pct}%", "height": "5px",
                    "background": color, "borderRadius": "3px",
                    "transition": "width 0.4s ease",
                }),
                style={"flex": "1", "background": "#F2F2F7",
                       "borderRadius": "3px", "height": "5px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
    ], style={"marginBottom": "7px"})
