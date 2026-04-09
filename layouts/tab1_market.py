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
            dbc.Col(_kpi_card("inv-kpi-listings",  "Listings shown",       "—",    "bi-house-door"),      md=3),
            dbc.Col(_kpi_card("inv-kpi-price",     "Avg nightly price",    "—",    "bi-currency-dollar"), md=3),
            dbc.Col(_kpi_card("inv-kpi-superhost", "Superhost %",          "—",    "bi-star"),            md=3),
            dbc.Col(_kpi_card("inv-kpi-sentiment", "Avg sentiment score",  "—",    "bi-chat-heart"),      md=3),
        ], className="mb-3 g-3"),

        # ── Main row: filters | map | detail card ─────────────────────────────
        dbc.Row([

            # Left: Filters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters", className="fw-semibold"),
                    dbc.CardBody([

                        dbc.Label("Neighbourhood", className="small fw-semibold text-muted"),
                        dcc.Dropdown(
                            id="mkt-neighbourhood",
                            options=[{"label": "All", "value": "All"}] +
                                    [{"label": n, "value": n} for n in neighbourhoods],
                            value="All", clearable=False, className="mb-3",
                        ),

                        dbc.Label("Room type", className="small fw-semibold text-muted"),
                        dcc.Dropdown(
                            id="mkt-room-type",
                            options=[{"label": "All", "value": "All"}] +
                                    [{"label": r, "value": r} for r in room_types],
                            value="All", clearable=False, className="mb-3",
                        ),

                        dbc.Label("Property type", className="small fw-semibold text-muted"),
                        dcc.Dropdown(
                            id="mkt-property-type",
                            options=[{"label": "All", "value": "All"}] +
                                    [{"label": p, "value": p} for p in property_types],
                            value="All", clearable=False, className="mb-3",
                        ),

                        dbc.Label("Price range ($/night)", className="small fw-semibold text-muted"),
                        dcc.RangeSlider(
                            id="mkt-price-range",
                            min=10, max=3000, step=10,
                            value=[10, 3000],
                            marks={10: "$10", 500: "$500", 1000: "$1k", 2000: "$2k", 3000: "$3k"},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-3",
                        ),

                        dbc.Label("Superhost", className="small fw-semibold text-muted"),
                        dbc.RadioItems(
                            id="mkt-superhost",
                            options=[
                                {"label": "All",        "value": "all"},
                                {"label": "Superhost",  "value": "yes"},
                                {"label": "Non-superhost", "value": "no"},
                            ],
                            value="all", inline=False, className="mb-4",
                        ),

                        dbc.Button(
                            [html.I(className="bi bi-arrow-counterclockwise me-1"), "Reset filters"],
                            id="mkt-reset-btn",
                            color="outline-secondary",
                            size="sm",
                            className="w-100",
                        ),
                    ]),
                ], className="shadow-sm h-100"),
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
                    ], className="p-1"),
                ], className="shadow-sm"),
            ], md=7),

            # Right: Detail card
            dbc.Col([
                html.Div(id="mkt-detail-panel", children=[_empty_detail_card()])
            ], md=3),

        ], className="g-3"),
    ], className="p-3")


# ── Sub-components ────────────────────────────────────────────────────────────

def _kpi_card(card_id, label, value, icon):
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.I(className=f"bi {icon} fs-3 text-primary"), width="auto"),
                dbc.Col([
                    html.P(label, className="text-muted small mb-0"),
                    html.H4(value, id=card_id, className="fw-bold mb-0"),
                ]),
            ], align="center"),
        ]),
    ], className="shadow-sm h-100")


def _empty_detail_card():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-cursor-fill fs-1 text-muted"),
                html.P("Click a listing on the map to see details.",
                       className="text-muted small mt-2 text-center"),
            ], className="d-flex flex-column align-items-center justify-content-center",
               style={"minHeight": "300px"}),
        ])
    ], className="shadow-sm h-100")


def build_detail_card(row: dict):
    """Build the listing detail card from a clicked map point."""

    superhost = row.get("host_is_superhost", 0)
    superhost_badge = dbc.Badge("⭐ Superhost", color="warning", className="me-1") if superhost else dbc.Badge("Non-superhost", color="secondary")

    price      = row.get("price_clean")
    price_str  = f"${price:,.0f} / night" if price and price == price else "N/A"

    sentiment  = row.get("sentiment_polarity_mean")
    sent_str   = f"{sentiment:.2f}" if sentiment == sentiment and sentiment is not None else "N/A"

    reviews    = int(row.get("number_of_reviews", 0))
    amenities  = int(row.get("amenity_count", 0))

    # Theme scores for radar
    themes = {
        "Cleanliness":    row.get("theme_cleanliness_mean"),
        "Communication":  row.get("theme_communication_mean"),
        "Check-in":       row.get("theme_checkin_mean"),
        "Location":       row.get("theme_location_mean"),
        "Amenities":      row.get("theme_amenities_mean"),
        "Value":          row.get("theme_value_mean"),
        "Comfort":        row.get("theme_comfort_mean"),
        "Accuracy":       row.get("theme_accuracy_mean"),
    }

    return dbc.Card([
        dbc.CardHeader([
            html.Span(row.get("name", "Listing"), className="fw-semibold small"),
        ]),
        dbc.CardBody([

            # Badges
            html.Div([superhost_badge], className="mb-2"),

            # Core info
            _detail_row("bi-geo-alt",      row.get("neighbourhood_cleansed", "—")),
            _detail_row("bi-house",        row.get("room_type", "—")),
            _detail_row("bi-building",     row.get("property_type", "—")),
            _detail_row("bi-currency-dollar", price_str),
            _detail_row("bi-chat-left-text", f"{reviews} reviews"),
            _detail_row("bi-emoji-smile",  f"Sentiment: {sent_str}"),
            _detail_row("bi-check2-all",   f"{amenities} amenities"),

            html.Hr(className="my-2"),

            # Review theme bars
            html.P("Review themes", className="small fw-semibold text-muted mb-2"),
            *[_theme_bar(label, score) for label, score in themes.items()
              if score is not None and score == score],

            # Link to listing
            html.Div(
                html.A("View on Airbnb →",
                       href=row.get("listing_url", "#"),
                       target="_blank",
                       className="small text-primary"),
                className="mt-2"
            ),
        ]),
    ], className="shadow-sm")


def _detail_row(icon, text):
    return dbc.Row([
        dbc.Col(html.I(className=f"bi {icon} text-muted"), width=1),
        dbc.Col(html.Span(str(text), className="small"), width=11),
    ], className="mb-1 align-items-center")


def _theme_bar(label: str, score: float):
    """Mini horizontal bar for a review theme score (0–1 range)."""
    pct   = min(int(score * 100), 100)
    color = "#198754" if score > 0.5 else "#ffc107" if score > 0.2 else "#dc3545"
    return html.Div([
        dbc.Row([
            dbc.Col(html.Span(label, className="text-muted", style={"fontSize": "11px"}), width=5),
            dbc.Col(
                html.Div(
                    html.Div(style={
                        "width": f"{pct}%", "height": "6px",
                        "background": color, "borderRadius": "3px",
                    }),
                    style={"background": "#e9ecef", "borderRadius": "3px"},
                ), width=7
            ),
        ], align="center", className="mb-1")
    ])
