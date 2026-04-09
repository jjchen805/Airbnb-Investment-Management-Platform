"""
Tab 2 — Superhost Advisor Layout
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


AMENITY_LABELS = {
    "has_wifi":            "WiFi",
    "has_kitchen":         "Kitchen",
    "has_washer":          "Washer",
    "has_dryer":           "Dryer",
    "has_parking":         "Parking",
    "has_air_conditioning":"Air conditioning",
    "has_heating":         "Heating",
    "has_tv":              "TV",
    "has_self_check-in":   "Self check-in",
    "has_coffee":          "Coffee maker",
    "has_hair_dryer":      "Hair dryer",
    "has_iron":            "Iron",
    "has_gym":             "Gym",
    "has_pool":            "Pool",
    "has_hot_tub":         "Hot tub",
    "has_elevator":        "Elevator",
}


def advisor_layout(df_listings, selected_listing_id=None):
    """
    df_listings : the full dashboard DataFrame (for populating dropdown)
    selected_listing_id : pre-selected from Tab 1 map click via dcc.Store
    """
    listing_options = [
        {"label": f"{row['name'][:45]}  —  {row['neighbourhood_top']}", "value": row["id"]}
        for _, row in df_listings[["id", "name", "neighbourhood_top"]]
            .dropna()
            .sort_values("neighbourhood_top")
            .iterrows()
    ]

    return html.Div([
        html.H4("Superhost Advisor", className="mt-3 mb-1 fw-semibold"),
        html.P(
            "Select an existing listing to see its Superhost potential, "
            "key weaknesses, and actionable recommendations.",
            className="text-muted mb-3"
        ),

        # Listing selector
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="bi bi-search")),
                    dcc.Dropdown(
                        id="adv-listing-select",
                        options=listing_options,
                        value=selected_listing_id,
                        placeholder="Search or select a listing...",
                        clearable=True,
                        style={"flex": "1"},
                    ),
                ]),
            ], md=8),
            dbc.Col([
                dbc.Alert(
                    [html.I(className="bi bi-info-circle me-2"),
                     "Or click a listing on the Map Explorer tab to auto-select it here."],
                    color="info", className="py-2 mb-0 small",
                ),
            ], md=4),
        ], className="mb-4 align-items-center"),

        # Output panel — populated by callback
        html.Div(id="adv-output-panel", children=[_empty_panel()]),

    ], className="p-3")


def _empty_panel():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-stars fs-1 text-muted"),
                html.P("Select a listing above to see its Superhost analysis.",
                       className="text-muted text-center mt-3"),
            ], className="d-flex flex-column align-items-center justify-content-center py-5")
        ])
    ], className="shadow-sm")


# ── Output panel builder ──────────────────────────────────────────────────────

def build_advisor_panel(row: dict, shap_drivers: list, recommendations: list, strengths: list, weaknesses: list):
    """
    row             : listing row as dict from dashboard_listings.csv
    shap_drivers    : list of {label, shap_val, direction} sorted by abs(shap_val)
    recommendations : list of {title, reason, impact} dicts
    strengths       : list of str
    weaknesses      : list of str
    """
    return dbc.Row([
        # Left column: overview + probability + strengths/weaknesses
        dbc.Col([
            _overview_card(row),
            _probability_card(row),
            _strengths_weaknesses_card(strengths, weaknesses),
        ], md=4),

        # Right column: recommendations + SHAP chart
        dbc.Col([
            _recommendations_card(recommendations),
            _shap_chart_card(shap_drivers, row),
        ], md=8),
    ], className="g-3")


def _overview_card(row: dict):
    superhost = row.get("host_is_superhost", 0)
    price     = row.get("price_clean")
    sentiment = row.get("sentiment_polarity_mean")
    response  = row.get("host_response_rate_clean")
    amenities = int(row.get("amenity_count", 0))
    has_pic   = row.get("host_has_profile_pic_num", 0)
    verified  = row.get("host_identity_verified_num", 0)

    # Key amenities present/missing
    key_amenities = ["has_wifi", "has_kitchen", "has_self_check-in",
                     "has_parking", "has_air_conditioning", "has_tv"]
    amenity_chips = []
    for col in key_amenities:
        label   = AMENITY_LABELS.get(col, col)
        present = row.get(col, 0)
        amenity_chips.append(
            dbc.Badge(
                [html.I(className=f"bi {'bi-check' if present else 'bi-x'} me-1"), label],
                color="success" if present else "danger",
                className="me-1 mb-1",
            )
        )

    return dbc.Card([
        dbc.CardHeader("📋 Listing Overview"),
        dbc.CardBody([
            html.H6(row.get("name", "—")[:50], className="fw-semibold mb-3"),

            _info_row("bi-star-fill",
                      "Superhost" if superhost else "Not a superhost",
                      "text-warning" if superhost else "text-muted"),
            _info_row("bi-currency-dollar",
                      f"${price:,.0f} / night" if price and price == price else "N/A"),
            _info_row("bi-chat-dots",
                      f"Response rate: {response:.0f}%" if response and response == response else "Response rate: N/A"),
            _info_row("bi-emoji-smile",
                      f"Sentiment: {sentiment:.2f}" if sentiment and sentiment == sentiment else "Sentiment: N/A"),
            _info_row("bi-person-check" if has_pic else "bi-person-x",
                      "Profile picture set" if has_pic else "No profile picture",
                      "text-success" if has_pic else "text-danger"),
            _info_row("bi-shield-check" if verified else "bi-shield-x",
                      "Identity verified" if verified else "Identity not verified",
                      "text-success" if verified else "text-danger"),

            html.Hr(className="my-2"),
            html.P(f"Amenities: {amenities} total", className="small text-muted mb-1"),
            html.Div(amenity_chips),
        ]),
    ], className="shadow-sm mb-3")


def _probability_card(row: dict):
    prob       = row.get("superhost_probability", 0.5)
    prob_pct   = int(prob * 100)
    is_sh      = row.get("host_is_superhost", 0)

    if prob >= 0.65:
        label, color, icon = "High potential",     "success", "bi-trophy-fill"
    elif prob >= 0.40:
        label, color, icon = "Moderate potential", "warning", "bi-graph-up"
    else:
        label, color, icon = "Low potential",      "danger",  "bi-exclamation-triangle"

    # If already a superhost, contextualise differently
    context = (
        "This listing is already a Superhost — the score reflects how strongly "
        "the model associates its features with Superhost status."
        if is_sh else
        "The model estimates this listing's likelihood of achieving Superhost status "
        "based on host setup, amenities, and review patterns."
    )

    return dbc.Card([
        dbc.CardHeader("⭐ Superhost Probability"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H1(f"{prob_pct}%",
                            className=f"display-4 fw-bold text-{color} mb-0",
                            style={"letterSpacing": "-2px"}),
                    dbc.Badge(
                        [html.I(className=f"bi {icon} me-1"), label],
                        color=color, className="mt-1"
                    ),
                ], width=5),
                dbc.Col([
                    dbc.Progress(
                        value=prob_pct,
                        color=color,
                        style={"height": "12px", "borderRadius": "6px"},
                        className="mb-2",
                    ),
                    html.P(context, className="small text-muted mb-0"),
                ], width=7),
            ], align="center"),
        ]),
    ], className="shadow-sm mb-3")


def _strengths_weaknesses_card(strengths: list, weaknesses: list):
    def make_items(items, icon, text_class):
        if not items:
            return html.P("None identified", className="text-muted small")
        return html.Ul([
            html.Li([html.I(className=f"bi {icon} me-2 {text_class}"), item],
                    className="small mb-1")
            for item in items
        ], className="ps-2 mb-0", style={"listStyle": "none"})

    return dbc.Card([
        dbc.CardHeader("🔍 Strengths vs Weaknesses"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Strengths", className="fw-semibold text-success small mb-2"),
                    make_items(strengths, "bi-check-circle-fill", "text-success"),
                ], width=6),
                dbc.Col([
                    html.P("Needs improvement", className="fw-semibold text-danger small mb-2"),
                    make_items(weaknesses, "bi-x-circle-fill", "text-danger"),
                ], width=6),
            ]),
        ]),
    ], className="shadow-sm")


def _recommendations_card(recommendations: list):
    if not recommendations:
        return dbc.Card([
            dbc.CardHeader("💡 Top Recommendations"),
            dbc.CardBody(html.P("No recommendations — this listing is already well-optimised.",
                                className="text-muted small")),
        ], className="shadow-sm mb-3")

    cards = []
    for i, rec in enumerate(recommendations):
        impact_color = {"high": "danger", "medium": "warning", "low": "secondary"}.get(
            rec.get("impact", "medium"), "secondary"
        )
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dbc.Badge(f"#{i+1}", color="primary",
                                      className="rounded-circle px-2"),
                            width="auto"
                        ),
                        dbc.Col([
                            html.Span(rec["title"], className="fw-semibold small"),
                            dbc.Badge(
                                f"{rec.get('impact','').title()} impact",
                                color=impact_color,
                                className="ms-2 small",
                            ),
                        ]),
                    ], align="center", className="mb-1"),
                    html.P(rec["reason"], className="small text-muted mb-0 mt-1"),
                ], className="py-2"),
            ], className="mb-2 border-start border-primary border-3")
        )

    return dbc.Card([
        dbc.CardHeader([
            "💡 Top Recommendations",
            html.Span(" — ranked by expected impact on Superhost probability",
                      className="text-muted fw-normal small"),
        ]),
        dbc.CardBody(cards),
    ], className="shadow-sm mb-3")


def _shap_chart_card(shap_drivers: list, row: dict):
    """
    Waterfall-style SHAP bar chart showing what pushed this listing
    toward or away from Superhost status.
    """
    if not shap_drivers:
        return html.Div()

    max_abs = max(abs(d["shap_val"]) for d in shap_drivers) or 1

    bars = []
    for d in shap_drivers:
        pct_width  = int(abs(d["shap_val"]) / max_abs * 100)
        color      = "#198754" if d["direction"] == "up" else "#dc3545"
        sign       = "+" if d["direction"] == "up" else "−"
        direction_label = "toward Superhost" if d["direction"] == "up" else "away from Superhost"

        bars.append(
            dbc.Row([
                dbc.Col(
                    html.Span(d["label"], className="text-muted",
                              style={"fontSize": "12px"}),
                    width=5
                ),
                dbc.Col(
                    html.Div(
                        html.Div(style={
                            "width": f"{pct_width}%", "height": "10px",
                            "background": color, "borderRadius": "4px",
                            "transition": "width 0.4s ease",
                        }),
                        style={"background": "#e9ecef", "borderRadius": "4px"},
                        title=f"{sign} pushes {direction_label}",
                    ),
                    width=5
                ),
                dbc.Col(
                    html.Span(sign,
                              style={"color": color, "fontSize": "14px", "fontWeight": "600"}),
                    width=2
                ),
            ], align="center", className="mb-2")
        )

    is_sh    = row.get("host_is_superhost", 0)
    subtitle = (
        "Factors that contributed most to this listing's Superhost classification."
        if is_sh else
        "Factors most influencing whether this listing could become a Superhost."
    )

    return dbc.Card([
        dbc.CardHeader([
            "📊 Model Explanation (SHAP)",
            html.Span(" — green pushes toward Superhost, red pushes away",
                      className="text-muted fw-normal small"),
        ]),
        dbc.CardBody([
            html.P(subtitle, className="small text-muted mb-3"),
            *bars,
            html.Hr(className="my-2"),
            html.P(
                "Computed using CatBoost TreeSHAP — values show each feature's "
                "contribution to this specific listing's prediction.",
                className="text-muted small mb-0"
            ),
        ]),
    ], className="shadow-sm")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _info_row(icon, text, text_class=""):
    return dbc.Row([
        dbc.Col(html.I(className=f"bi {icon} text-muted"), width=1),
        dbc.Col(html.Span(str(text), className=f"small {text_class}"), width=11),
    ], className="mb-1 align-items-center")
