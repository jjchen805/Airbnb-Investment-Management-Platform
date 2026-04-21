"""
Tab 1 Callbacks — Market Explorer
==================================
Handles:
  - City selection → reload correct dataframe
  - Filter changes → map update + KPI cards
  - Map click → listing detail card
  - Reset button → clear all filters
  - Syncs selected listing id to dcc.Store for cross-tab use
"""
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
 
from dash import Input, Output, State, no_update
from layouts.tab1_market import build_detail_card
 
# ── Load all city dataframes at startup ───────────────────────────────────────
# Each city's dashboard_listings_{city}.csv is loaded once into memory.
# The city selector in the navbar switches between them.
CITIES = [
    c.replace("dashboard_listings_", "").replace(".csv", "")
    for c in os.listdir("data")
    if c.startswith("dashboard_listings_") and c.endswith(".csv")
]
 
ALL_DF = {}
for city in CITIES:
    df = pd.read_csv(f"data/dashboard_listings_{city}.csv")
    df["host_is_superhost"]       = pd.to_numeric(df["host_is_superhost"],       errors="coerce").fillna(0)
    df["price_clean"]             = pd.to_numeric(df["price_clean"],             errors="coerce")
    df["sentiment_polarity_mean"] = pd.to_numeric(df["sentiment_polarity_mean"], errors="coerce")
    ALL_DF[city] = df
 
print(f"  Market Explorer: loaded {len(CITIES)} cities — {CITIES}")
 
COLOR_SUPERHOST     = "#0071E3"   # Apple blue — premium, positive
COLOR_NON_SUPERHOST = "#B0BEC5"   # soft cool slate — recessive
 
 
def _get_df(city: str) -> pd.DataFrame:
    return ALL_DF.get(city, next(iter(ALL_DF.values())))
 
 
def _apply_filters(dff, neighbourhood, room_type, property_type, price_range, superhost_filter):
    if neighbourhood and neighbourhood != "All":
        dff = dff[dff["neighbourhood_top"] == neighbourhood]
    if room_type and room_type != "All":
        dff = dff[dff["room_type"] == room_type]
    if property_type and property_type != "All":
        dff = dff[dff["property_type_simple"] == property_type]
    if price_range:
        dff = dff[
            (dff["price_clean"] >= price_range[0]) &
            (dff["price_clean"] <= price_range[1])
        ]
    if superhost_filter == "yes":
        dff = dff[dff["host_is_superhost"] == 1]
    elif superhost_filter == "no":
        dff = dff[dff["host_is_superhost"] == 0]
    return dff
 
 
def _build_map(dff: pd.DataFrame):
    sh  = dff[dff["host_is_superhost"] == 1]
    nsh = dff[dff["host_is_superhost"] == 0]
 
    def make_trace(subset, name, color):
        sizes = subset["price_clean"].clip(upper=800).fillna(100)
        sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min() + 1)) * 10 + 6
 
        hover = (
            "<span style='font-size:14px;font-weight:600;color:#1D1D1F'>"
            + subset["name"].fillna("Listing").str[:38] + "</span><br>"
            + "<span style='color:#AEAEB2;font-size:12px'>Neighbourhood</span> "
            + "<span style='color:#3A3A3C;font-size:12px'>"
            + subset["neighbourhood_top"].fillna("—") + "</span><br>"
            + "<span style='color:#AEAEB2;font-size:12px'>Room type</span> "
            + "<span style='color:#3A3A3C;font-size:12px'>"
            + subset["room_type"].fillna("—") + "</span><br>"
            + "<span style='color:#AEAEB2;font-size:12px'>Price</span> "
            + "<span style='color:#1D1D1F;font-size:13px;font-weight:500'>$"
            + subset["price_clean"].fillna(0).astype(int).astype(str) + " / night</span><br>"
            + "<span style='color:#AEAEB2;font-size:12px'>Reviews</span> "
            + "<span style='color:#3A3A3C;font-size:12px'>"
            + subset["number_of_reviews"].fillna(0).astype(int).astype(str) + "</span>"
            + "<extra></extra>"
        )
 
        return go.Scattermapbox(
            lat=subset["latitude"],
            lon=subset["longitude"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.55,
                sizemode="diameter",
            ),
            text=subset["name"].fillna(""),
            customdata=subset.index.tolist(),
            hovertemplate=hover,
            name=name,
        )
 
    fig = go.Figure()
    fig.add_trace(make_trace(nsh, "Non-superhost", COLOR_NON_SUPERHOST))
    fig.add_trace(make_trace(sh,  "Superhost",     COLOR_SUPERHOST))
 
    center_lat = dff["latitude"].mean()  if len(dff) > 0 else 37.7749
    center_lon = dff["longitude"].mean() if len(dff) > 0 else -122.4194
 
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11.5,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.01,
            xanchor="right",  x=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(size=12),
        ),
        uirevision="map",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E5E5EA",
            font=dict(family="-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                      size=13, color="#1D1D1F"),
        ),
    )
    return fig
 
 
def _build_kpis(dff: pd.DataFrame):
    n_listings    = f"{len(dff):,}"
    valid_prices  = dff["price_clean"].dropna()
    avg_price     = f"${valid_prices.mean():,.0f}" if len(valid_prices) > 0 else "—"
    superhost_pct = f"{dff['host_is_superhost'].mean() * 100:.1f}%" if len(dff) > 0 else "—"
    valid_sent    = dff["sentiment_polarity_mean"].dropna()
    avg_sentiment = f"{valid_sent.mean():.2f}" if len(valid_sent) > 0 else "—"
    return n_listings, avg_price, superhost_pct, avg_sentiment
 
 
def register_market_callbacks(app):
 
    # ── Map + KPIs — reacts to city change and all filters ────────────────
    @app.callback(
        Output("mkt-map",           "figure"),
        Output("inv-kpi-listings",  "children"),
        Output("inv-kpi-price",     "children"),
        Output("inv-kpi-superhost", "children"),
        Output("inv-kpi-sentiment", "children"),
        Input("selected-city",      "data"),
        Input("mkt-neighbourhood",  "value"),
        Input("mkt-room-type",      "value"),
        Input("mkt-property-type",  "value"),
        Input("mkt-price-range",    "value"),
        Input("mkt-superhost",      "value"),
    )
    def update_map_and_kpis(city, neighbourhood, room_type, property_type, price_range, superhost_filter):
        dff  = _get_df(city).copy()
        dff  = _apply_filters(dff, neighbourhood, room_type, property_type, price_range, superhost_filter)
        fig  = _build_map(dff)
        kpis = _build_kpis(dff)
        return fig, *kpis
 
    # ── Detail card on map click ───────────────────────────────────────────
    @app.callback(
        Output("mkt-detail-panel",    "children"),
        Output("selected-listing-id", "data"),
        Input("mkt-map",              "clickData"),
        State("selected-city",        "data"),
        prevent_initial_call=True,
    )
    def show_listing_detail(click_data, city):
        if not click_data:
            return no_update, no_update
 
        df       = _get_df(city)
        point    = click_data["points"][0]
        df_index = point.get("customdata")
 
        if df_index is None or df_index not in df.index:
            return no_update, no_update
 
        row        = df.loc[df_index].to_dict()
        listing_id = row.get("id")
        return build_detail_card(row), listing_id
 
    # ── Reset filters ──────────────────────────────────────────────────────
    @app.callback(
        Output("mkt-neighbourhood", "value"),
        Output("mkt-room-type",     "value"),
        Output("mkt-property-type", "value"),
        Output("mkt-price-range",   "value"),
        Output("mkt-superhost",     "value"),
        Input("mkt-reset-btn",      "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_filters(_):
        return "All", "All", "All", [10, 3000], "all"
