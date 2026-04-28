"""
Tab 2 — Superhost Advisor Layout
"""
from dash import html, dcc
import pandas as pd
import dash_bootstrap_components as dbc


AMENITY_LABELS = {
    "has_wifi":            "WiFi",
    "has_kitchen":         "Kitchen",
    "has_washer":          "Washer",
    "has_dryer":           "Dryer",
    "has_parking":         "Parking",
    "has_air_conditioning":"AC",
    "has_heating":         "Heating",
    "has_tv":              "TV",
    "has_self_check-in":   "Self check-in",
    "has_coffee":          "Coffee",
    "has_hair_dryer":      "Hair dryer",
    "has_iron":            "Iron",
    "has_gym":             "Gym",
    "has_pool":            "Pool",
    "has_hot_tub":         "Hot tub",
    "has_elevator":        "Elevator",
}

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


def advisor_layout(df_listings, selected_listing_id=None):
    # Neighbourhood options for filter dropdown
    neighbourhoods = sorted(df_listings["neighbourhood_top"].dropna().unique().tolist())
    neighbourhood_options = [{"label": "All neighbourhoods", "value": "__all__"}] + [
        {"label": n, "value": n} for n in neighbourhoods
    ]

    # Full listing options (neighbourhood filter narrows these via callback)
    listing_options = _build_listing_options(df_listings, neighbourhood=None)

    return html.Div([
        # Page header
        html.Div([
            html.H4("Superhost Advisor",
                    style={"fontSize": "22px", "fontWeight": "700",
                           "color": C["gray1"], "marginBottom": "4px"}),
            html.P("Filter by neighborhood, then search for a listing by name, address, or ID.",
                   style={"fontSize": "14px", "color": C["gray3"], "margin": 0}),
        ], style={"marginBottom": "24px"}),

        # Filters row
        dbc.Row([
            # Neighbourhood filter
            dbc.Col([
                html.Label("Neighborhood",
                           style={"fontSize": "12px", "fontWeight": "600",
                                  "color": C["gray3"], "marginBottom": "6px",
                                  "display": "block", "letterSpacing": "0.4px"}),
                dcc.Dropdown(
                    id="adv-neighbourhood-filter",
                    options=neighbourhood_options,
                    value="__all__",
                    clearable=False,
                    searchable=True,
                    placeholder="All neighborhoods",
                    style={"fontSize": "14px"},
                ),
            ], md=4),

            # Listing search
            dbc.Col([
                html.Label("Listing",
                           style={"fontSize": "12px", "fontWeight": "600",
                                  "color": C["gray3"], "marginBottom": "6px",
                                  "display": "block", "letterSpacing": "0.4px"}),
                dbc.InputGroup([
                    dbc.InputGroupText(
                        html.I(className="bi bi-search",
                               style={"color": C["gray4"], "fontSize": "13px"}),
                        style={"background": "#fff", "border": "1px solid #D1D1D6",
                               "borderRight": "none", "borderRadius": "10px 0 0 10px"}
                    ),
                    dcc.Dropdown(
                        id="adv-listing-select",
                        options=listing_options,
                        value=selected_listing_id,
                        placeholder="Search by name, neighborhood, or price...",
                        clearable=True,
                        searchable=True,
                        style={"flex": "1", "fontSize": "14px"},
                    ),
                ]),
            ], md=5),

            # Map click hint
            dbc.Col([
                html.Div(style={"height": "24px"}),  # spacer to align with inputs
                html.Div([
                    html.I(className="bi bi-info-circle",
                           style={"color": C["blue"], "marginRight": "8px",
                                  "fontSize": "13px"}),
                    html.Span("Or click a listing on the Market Explorer map.",
                              style={"fontSize": "13px", "color": C["gray3"]}),
                ], style={
                    "background": "#EBF5FB", "borderRadius": "10px",
                    "padding": "10px 14px", "display": "flex", "alignItems": "center",
                    "height": "38px",
                }),
            ], md=3),
        ], className="mb-4 align-items-end g-3"),

        # Listing count badge
        html.Div(id="adv-listing-count",
                 style={"fontSize": "12px", "color": C["gray4"],
                        "marginBottom": "20px"}),

        # Output panel
        html.Div(id="adv-output-panel", children=[_empty_panel()]),

    ], style={"padding": "24px"})


def _build_listing_options(df, neighbourhood=None):
    """Build listing dropdown options, optionally filtered by neighbourhood."""
    df_filt = df.copy()
    if neighbourhood and neighbourhood != "__all__":
        df_filt = df_filt[df_filt["neighbourhood_top"] == neighbourhood]

    df_filt = (df_filt[["id", "name", "neighbourhood_top", "price_clean",
                          "host_is_superhost", "room_type"]]
               .dropna(subset=["id", "name"])
               .sort_values(["neighbourhood_top", "name"]))

    options = []
    for _, row in df_filt.iterrows():
        name   = str(row["name"])[:40]
        nbhd   = row.get("neighbourhood_top", "")
        price  = row.get("price_clean")
        is_sh  = row.get("host_is_superhost", 0)
        star   = " ⭐" if is_sh else ""
        price_str = f" · ${price:,.0f}/night" if pd.notna(price) else ""
        label  = f"{name}{star}  —  {nbhd}{price_str}"
        options.append({"label": label, "value": row["id"]})

    return options


def _empty_panel():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-stars",
                       style={"fontSize": "32px", "color": C["gray5"]}),
                html.P("Select a listing above to see its Superhost analysis.",
                       style={"fontSize": "14px", "color": C["gray4"],
                              "textAlign": "center", "margin": "12px 0 0"}),
            ], style={
                "display": "flex", "flexDirection": "column",
                "alignItems": "center", "justifyContent": "center",
                "padding": "60px 0",
            }),
        ]),
    ])


def build_advisor_panel(row, shap_drivers, recommendations, strengths, weaknesses):
    return dbc.Row([
        dbc.Col([
            _overview_card(row),
            _probability_card(row),
            _strengths_weaknesses_card(strengths, weaknesses),
        ], md=4),
        dbc.Col([
            _recommendations_card(recommendations),
            _shap_chart_card(shap_drivers, row),
            _advisor_agent_panel(),
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

    key_amenities = ["has_wifi", "has_kitchen", "has_self_check-in",
                     "has_parking", "has_air_conditioning", "has_tv"]

    def amenity_pill(col):
        label   = AMENITY_LABELS.get(col, col)
        present = row.get(col, 0)
        return html.Span(
            [html.I(className=f"bi {'bi-check' if present else 'bi-x'}",
                    style={"marginRight": "3px"}), label],
            style={
                "fontSize": "11px", "fontWeight": "500",
                "padding": "3px 8px", "borderRadius": "6px",
                "marginRight": "4px", "marginBottom": "4px",
                "display": "inline-block",
                "background": "#D1F5D3" if present else "#FFE5E5",
                "color": "#1A7431" if present else "#9B1B1B",
            }
        )

    return dbc.Card([
        dbc.CardBody([
            html.P(row.get("name", "—")[:50],
                   style={"fontSize": "15px", "fontWeight": "600",
                          "color": C["gray1"], "marginBottom": "16px",
                          "lineHeight": "1.3"}),

            *[_info_row(icon, text, accent) for icon, text, accent in [
                ("bi-star-fill",
                 "Superhost" if superhost else "Not a superhost",
                 C["orange"] if superhost else C["gray4"]),
                ("bi-currency-dollar",
                 f"${price:,.0f} / night" if price and price == price else "N/A",
                 C["gray3"]),
                ("bi-chat-dots",
                 f"Response rate: {response:.0f}%" if response and response == response else "Response rate: N/A",
                 C["gray3"]),
                ("bi-emoji-smile",
                 f"Sentiment: {sentiment:.2f}" if sentiment and sentiment == sentiment else "Sentiment: N/A",
                 C["gray3"]),
                ("bi-person-check" if has_pic else "bi-person-x",
                 "Profile picture set" if has_pic else "No profile picture",
                 C["green"] if has_pic else C["red"]),
                ("bi-shield-check" if verified else "bi-shield-x",
                 "Identity verified" if verified else "Identity not verified",
                 C["green"] if verified else C["red"]),
            ]],

            html.Hr(style={"margin": "14px 0"}),
            html.P(f"{amenities} amenities",
                   style={"fontSize": "11px", "fontWeight": "600",
                          "color": C["gray4"], "textTransform": "uppercase",
                          "letterSpacing": "0.5px", "marginBottom": "8px"}),
            html.Div([amenity_pill(col) for col in key_amenities]),
        ]),
    ], className="mb-3")


def _probability_card(row: dict):
    prob     = row.get("superhost_probability", 0.5)
    prob_pct = int(prob * 100)
    is_sh    = row.get("host_is_superhost", 0)

    if prob >= 0.65:
        label, bar_color, text_color = "High potential",     C["green"],  "#1A7431"
    elif prob >= 0.40:
        label, bar_color, text_color = "Moderate potential", C["orange"], "#92600A"
    else:
        label, bar_color, text_color = "Low potential",      C["red"],    "#9B1B1B"

    context = (
        "Already a Superhost — score reflects model confidence in this status."
        if is_sh else
        "Estimated likelihood of achieving Superhost status."
    )

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(f"{prob_pct}%",
                          style={"fontSize": "48px", "fontWeight": "700",
                                 "color": bar_color, "letterSpacing": "-2px",
                                 "lineHeight": "1"}),
                html.Span(label,
                          style={"fontSize": "12px", "fontWeight": "600",
                                 "color": text_color, "background": f"{bar_color}18",
                                 "padding": "3px 10px", "borderRadius": "6px",
                                 "marginLeft": "10px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
            html.Div(
                html.Div(style={
                    "width": f"{prob_pct}%",
                    "height": "6px",
                    "background": bar_color,
                    "borderRadius": "3px",
                    "transition": "width 0.5s ease",
                }),
                style={"background": C["gray6"], "borderRadius": "3px",
                       "height": "6px", "marginBottom": "12px"},
            ),
            html.P(context,
                   style={"fontSize": "12px", "color": C["gray3"], "margin": 0}),
        ]),
    ], className="mb-3")


def _strengths_weaknesses_card(strengths, weaknesses):
    def items(lst, color, icon):
        if not lst:
            return html.P("None identified",
                          style={"fontSize": "13px", "color": C["gray4"]})
        return html.Div([
            html.Div([
                html.I(className=f"bi {icon}",
                       style={"color": color, "fontSize": "12px",
                              "marginRight": "6px", "flexShrink": "0"}),
                html.Span(item, style={"fontSize": "13px", "color": C["gray2"]}),
            ], style={"display": "flex", "alignItems": "flex-start",
                      "marginBottom": "6px"})
            for item in lst
        ])

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Strengths",
                           style={"fontSize": "11px", "fontWeight": "600",
                                  "color": C["green"], "textTransform": "uppercase",
                                  "letterSpacing": "0.5px", "marginBottom": "10px"}),
                    items(strengths, C["green"], "bi-check-circle-fill"),
                ], width=6),
                dbc.Col([
                    html.P("Needs work",
                           style={"fontSize": "11px", "fontWeight": "600",
                                  "color": C["red"], "textTransform": "uppercase",
                                  "letterSpacing": "0.5px", "marginBottom": "10px"}),
                    items(weaknesses, C["red"], "bi-x-circle-fill"),
                ], width=6),
            ]),
        ]),
    ])


def _recommendations_card(recommendations):
    if not recommendations:
        return dbc.Card([
            dbc.CardBody(
                html.P("No recommendations — this listing is already well-optimised.",
                       style={"fontSize": "13px", "color": C["gray3"]})
            ),
        ], className="mb-3")

    impact_colors = {
        "high":   (C["red"],    "#FFE5E5"),
        "medium": (C["orange"], "#FFF3D6"),
        "low":    (C["gray4"],  C["gray6"]),
    }

    items = []
    for i, rec in enumerate(recommendations):
        fg, bg = impact_colors.get(rec.get("impact", "low"), (C["gray4"], C["gray6"]))
        items.append(
            html.Div([
                html.Div([
                    html.Span(f"{i+1}",
                              style={"fontSize": "11px", "fontWeight": "700",
                                     "color": C["blue"], "background": "#E8F1FB",
                                     "width": "22px", "height": "22px",
                                     "borderRadius": "50%", "display": "flex",
                                     "alignItems": "center", "justifyContent": "center",
                                     "flexShrink": "0"}),
                    html.Div([
                        html.Div([
                            html.Span(rec["title"],
                                      style={"fontSize": "14px", "fontWeight": "600",
                                             "color": C["gray1"]}),
                            html.Span(rec.get("impact", "").title(),
                                      style={"fontSize": "10px", "fontWeight": "600",
                                             "color": fg, "background": bg,
                                             "padding": "2px 7px", "borderRadius": "4px",
                                             "marginLeft": "8px"}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "marginBottom": "4px"}),
                        html.P(rec["reason"],
                               style={"fontSize": "13px", "color": C["gray3"],
                                      "margin": 0, "lineHeight": "1.5"}),
                    ]),
                ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start"}),
            ], style={
                "padding": "16px 20px",
                "borderLeft": f"3px solid {C['blue']}",
                "borderRadius": "12px",
                "background": "#FAFAFA",
                "marginBottom": "10px",
            })
        )

    return dbc.Card([
        dbc.CardBody([
            html.P("TOP RECOMMENDATIONS",
                   style={"fontSize": "11px", "fontWeight": "600",
                          "color": C["gray4"], "letterSpacing": "0.8px",
                          "marginBottom": "16px"}),
            *items,
        ]),
    ], className="mb-3")


def _shap_chart_card(shap_drivers, row):
    if not shap_drivers:
        return html.Div()

    max_abs = max(abs(d["shap_val"]) for d in shap_drivers) or 1
    is_sh   = row.get("host_is_superhost", 0)

    bars = []
    for d in shap_drivers:
        pct   = int(abs(d["shap_val"]) / max_abs * 100)
        color = C["green"] if d["direction"] == "up" else C["red"]
        sign  = "+" if d["direction"] == "up" else "−"

        bars.append(html.Div([
            html.Span(d["label"],
                      style={"fontSize": "12px", "color": C["gray3"],
                             "width": "160px", "flexShrink": "0"}),
            html.Div(
                html.Div(style={
                    "width": f"{pct}%", "height": "8px",
                    "background": color, "borderRadius": "4px",
                    "transition": "width 0.4s ease",
                }),
                style={"flex": "1", "background": C["gray6"],
                       "borderRadius": "4px", "height": "8px"},
            ),
            html.Span(sign, style={"fontSize": "14px", "fontWeight": "700",
                                   "color": color, "width": "16px",
                                   "textAlign": "center", "flexShrink": "0"}),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "12px", "marginBottom": "10px"}))

    subtitle = (
        "Features that contributed most to this listing's Superhost classification."
        if is_sh else
        "Features most influencing this listing's Superhost probability."
    )

    return dbc.Card([
        dbc.CardBody([
            html.P("MODEL EXPLANATION (SHAP)",
                   style={"fontSize": "11px", "fontWeight": "600",
                          "color": C["gray4"], "letterSpacing": "0.8px",
                          "marginBottom": "4px"}),
            html.P(subtitle,
                   style={"fontSize": "13px", "color": C["gray3"],
                          "marginBottom": "20px"}),
            *bars,
            html.P(
                "Computed using CatBoost TreeSHAP. Green = pushes toward Superhost, "
                "red = pushes away.",
                style={"fontSize": "11px", "color": C["gray4"], "margin": "12px 0 0"}
            ),
        ]),
    ])


def _info_row(icon, text, accent):
    return html.Div([
        html.I(className=f"bi {icon}",
               style={"fontSize": "13px", "color": accent,
                      "width": "18px", "marginRight": "8px", "flexShrink": "0"}),
        html.Span(str(text), style={"fontSize": "13px", "color": C["gray2"]}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "7px"})

def _advisor_agent_panel():
    """AI Q&A + action plan panel for the Superhost Advisor."""
    from dash import html, dcc
    import pandas as pd
    import dash_bootstrap_components as dbc
 
    C = {
        "blue":  "#0071E3",
        "gray1": "#1D1D1F",
        "gray3": "#6E6E73",
        "gray4": "#AEAEB2",
        "gray6": "#F2F2F7",
    }
 
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("AI ADVISOR", style={
                        "fontSize": "11px", "fontWeight": "600",
                        "color": C["gray4"], "letterSpacing": "0.8px",
                        "marginBottom": "4px",
                    }),
                    html.P(
                        "Ask a question about this listing or generate a 7/30-day action plan.",
                        style={"fontSize": "13px", "color": C["gray3"], "marginBottom": "16px"},
                    ),
 
                    # Status text
                    html.P(id="adv-agent-status",
                           style={"fontSize": "12px", "color": C["gray4"],
                                  "marginBottom": "12px"}),
 
                    # Question input
                    dbc.Textarea(
                        id="adv-agent-question",
                        placeholder="e.g. How can I improve my check-in score?",
                        disabled=True,
                        style={
                            "fontSize": "13px", "borderRadius": "10px",
                            "border": "1px solid #D1D1D6",
                            "padding": "10px 12px", "resize": "none",
                            "height": "72px", "marginBottom": "10px",
                            "width": "100%",
                        },
                    ),
 
                    # Buttons
                    dbc.Row([
                        dbc.Col(
                            dbc.Button("Ask", id="adv-agent-ask-btn",
                                       disabled=True, n_clicks=0,
                                       style={"background": C["blue"], "border": "none",
                                              "borderRadius": "10px", "fontSize": "13px",
                                              "fontWeight": "500", "width": "100%"}),
                            width=4
                        ),
                        dbc.Col(
                            dbc.Button("Generate action plan",
                                       id="adv-agent-generate-btn",
                                       disabled=True, n_clicks=0,
                                       style={"background": "#FFFFFF", "border": "1px solid #D1D1D6",
                                              "borderRadius": "10px", "fontSize": "13px",
                                              "fontWeight": "500",
                                              "width": "100%"}),
                            width=8
                        ),
                    ], className="mb-3 g-2"),
 
                    # Response area
                    dcc.Loading(
                        dcc.Markdown(id="adv-agent-content",
                            style={"fontSize": "13px", "color": C["gray1"],
                                    "lineHeight": "1.6"},
                            className="markdown-body"),
                    ),
                ]),
            ]),
        ], md=12),
    ], className="g-3 mt-1")