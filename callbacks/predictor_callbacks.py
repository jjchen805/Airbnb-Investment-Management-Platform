"""
Tab 3 Callbacks — Investor Price Predictor
==========================================
Uses CatBoost's native TreeSHAP to produce local, per-row explanations.
Swaps price model and meta when the city selector changes.
"""
import json
import joblib
import os
import numpy as np
import pandas as pd
from catboost import Pool

from dash import Input, Output, State
from layouts.tab3_predictor import build_output_panel

# ── Load all city price models at startup ─────────────────────────────────────
CITIES = [
    c.replace("dashboard_meta_", "").replace(".json", "")
    for c in os.listdir("data")
    if c.startswith("dashboard_meta_") and c.endswith(".json")
]

ALL_PRICE_MODELS = {}
for city in CITIES:
    ALL_PRICE_MODELS[city] = {
        "model":    joblib.load(f"models/{city}/price_model.pkl"),
        "features": joblib.load(f"models/{city}/price_features.pkl"),
        "meta":     joblib.load(f"models/{city}/price_meta.pkl"),
    }

ALL_META = {}
for city in CITIES:
    with open(f"data/dashboard_meta_{city}.json") as f:
        ALL_META[city] = json.load(f)

print(f"  Predictor: loaded {len(CITIES)} cities — {CITIES}")

# ── Constants ─────────────────────────────────────────────────────────────────
AMENITY_COLS = [
    "has_wifi", "has_kitchen", "has_washer", "has_dryer", "has_parking",
    "has_air_conditioning", "has_heating", "has_tv", "has_self_check-in",
    "has_coffee", "has_hair_dryer", "has_iron", "has_gym", "has_pool",
    "has_hot_tub", "has_elevator",
]

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

FEATURE_HUMAN_LABELS = {
    "latitude":                          "Location",
    "longitude":                         "Location",
    "accommodates":                      "Guests accommodated",
    "bedrooms":                          "Bedrooms",
    "bathrooms_clean":                   "Bathrooms",
    "beds":                              "Beds",
    "amenity_count":                     "Total amenities",
    "is_entire_home":                    "Entire home",
    "is_private_room":                   "Private room",
    "neighbourhood_top_median_price":    "Neighbourhood price level",
    "property_type_simple_median_price": "Property type price level",
    "availability_365":                  "Annual availability",
    "minimum_nights":                    "Minimum nights",
    "host_is_superhost":                 "Superhost status",
    "instant_bookable_num":              "Instant bookable",
    "calculated_host_listings_count":    "Host listing count",
    **{col: AMENITY_LABELS[col] for col in AMENITY_COLS},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_city_assets(city: str):
    city = city if city in CITIES else CITIES[0]
    pm   = ALL_PRICE_MODELS[city]
    meta = ALL_META[city]
    return pm["model"], pm["features"], pm["meta"], meta


def _infer_room_flags(property_type: str):
    pt = property_type.lower()
    if "private room" in pt:
        return 0, 1, 0
    elif "shared room" in pt or "hotel" in pt:
        return 0, 0, 1
    return 1, 0, 0


def _build_feature_row(form_values: dict, price_features: list,
                       group_medians: dict, fillna_medians: dict,
                       nbhd_coords: dict) -> pd.DataFrame:
    neighbourhood = form_values["neighbourhood"]
    property_type = form_values["property_type"]
    lat, lon      = nbhd_coords.get(neighbourhood, (40.7128, -74.0060))
    is_entire, is_priv, is_shared = _infer_room_flags(property_type)

    amenity_vals = {col: (1 if form_values.get(col, 0) else 0) for col in AMENITY_COLS}

    row = {
        "accommodates":                      form_values.get("accommodates", 2),
        "bathrooms_clean":                   form_values.get("bathrooms_clean", 1.0),
        "bedrooms":                          form_values.get("bedrooms", 1),
        "beds":                              form_values.get("beds", 1),
        "minimum_nights":                    form_values.get("minimum_nights", 2),
        "maximum_nights":                    form_values.get("maximum_nights", 365),
        "availability_60":                   fillna_medians.get("availability_60", 12),
        "availability_365":                  form_values.get("availability_365", 200),
        "instant_bookable_num":              form_values.get("instant_bookable_num", 0),
        "host_is_superhost":                 form_values.get("host_is_superhost", 0.0),
        "calculated_host_listings_count":    form_values.get("host_listings", 1),
        "amenity_count":                     form_values.get("amenity_count", 20),
        "is_entire_home":                    is_entire,
        "is_private_room":                   is_priv,
        "is_shared_room":                    is_shared,
        "latitude":                          lat,
        "longitude":                         lon,
        "neighbourhood_top_median_price":    group_medians["neighbourhood_top"].get(neighbourhood, 169.0),
        "property_type_simple_median_price": group_medians["property_type_simple"].get(property_type, 169.0),
        **amenity_vals,
    }

    X_df = pd.DataFrame([row])

    for col_name in price_features:
        if col_name.startswith("neighbourhood_top_") and col_name != "neighbourhood_top_median_price":
            X_df[col_name] = 1 if neighbourhood == col_name.replace("neighbourhood_top_", "") else 0
        elif col_name.startswith("property_type_simple_") and col_name != "property_type_simple_median_price":
            X_df[col_name] = 1 if property_type == col_name.replace("property_type_simple_", "") else 0

    X_df = X_df.reindex(columns=price_features, fill_value=0)
    X_df = X_df.fillna(fillna_medians)
    return X_df


def _compute_shap(X_df: pd.DataFrame, price_model, price_features: list,
                  predicted_price: float) -> list:
    pool        = Pool(X_df)
    shap_matrix = price_model.get_feature_importance(data=pool, type="ShapValues")
    shap_vals   = dict(zip(price_features, shap_matrix[0, :-1]))

    loc_shap = shap_vals.pop("latitude", 0) + shap_vals.pop("longitude", 0)
    shap_vals["Location"] = loc_shap

    sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)

    drivers, seen = [], set()
    for feat, shap_val in sorted_shap:
        label = FEATURE_HUMAN_LABELS.get(feat, feat.replace("_", " ").title())
        if label in seen:
            continue
        seen.add(label)
        dollar_impact = predicted_price * float(np.expm1(shap_val))
        drivers.append({
            "label":         label,
            "shap_val":      float(shap_val),
            "dollar_impact": dollar_impact,
            "direction":     "up" if shap_val > 0 else "down",
        })
        if len(drivers) >= 8:
            break
    return drivers


def _get_amenity_lift(amenity_col: str, neighbourhood: str, meta: dict) -> float:
    amenity_lifts     = meta.get("amenity_lifts", {})
    amenity_lifts_by_nbhd = meta.get("amenity_lifts_by_nbhd", {})
    nbhd_data = amenity_lifts_by_nbhd.get(neighbourhood, {})
    val = nbhd_data.get(amenity_col)
    if val is not None:
        return float(val)
    return float(amenity_lifts.get(amenity_col, 0))


def predict_price(form_values: dict, city: str) -> dict:
    price_model, price_features, price_meta_city, meta = _get_city_assets(city)

    group_medians  = price_meta_city["group_medians"]
    fillna_medians = price_meta_city["fillna_medians"]
    nbhd_coords    = {r["neighbourhood_top"]: (r["latitude"], r["longitude"])
                      for r in meta["neighbourhoods"]}

    X_df = _build_feature_row(form_values, price_features, group_medians,
                               fillna_medians, nbhd_coords)

    y_log_pred      = price_model.predict(X_df)[0]
    predicted_price = float(np.expm1(y_log_pred))

    drivers = _compute_shap(X_df, price_model, price_features, predicted_price)

    neighbourhood = form_values["neighbourhood"]
    unchecked     = [col for col in AMENITY_COLS if not form_values.get(col, 0)]

    amenity_gaps = []
    for col in unchecked:
        market_lift = _get_amenity_lift(col, neighbourhood, meta)
        if market_lift > 10:
            amenity_gaps.append({
                "col":         col,
                "label":       AMENITY_LABELS[col],
                "market_lift": market_lift,
            })
    amenity_gaps = sorted(amenity_gaps, key=lambda x: x["market_lift"], reverse=True)[:4]

    nbhd_medians = {r["neighbourhood_top"]: r["median_price"] for r in meta["neighbourhoods"]}
    prop_medians = {r["property_type_simple"]: r["median_price"] for r in meta["property_types"]}

    nbhd_median = nbhd_medians.get(neighbourhood, meta["overall_median_price"])
    prop_median = prop_medians.get(form_values["property_type"], meta["overall_median_price"])
    pct_vs_nbhd = ((predicted_price - nbhd_median) / nbhd_median) * 100

    return {
        "predicted_price": predicted_price,
        "drivers":         drivers,
        "amenity_gaps":    amenity_gaps,
        "nbhd_median":     nbhd_median,
        "prop_median":     prop_median,
        "pct_vs_nbhd":     pct_vs_nbhd,
        "neighbourhood":   neighbourhood,
        "property_type":   form_values["property_type"],
        "city":            city, 
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_predictor_callbacks(app):

    @app.callback(
        Output("inv-output-panel",    "children"),
        Input("inv-predict-btn",      "n_clicks"),
        State("selected-city",        "data"),
        State("inv-neighbourhood",    "value"),
        State("inv-property-type",    "value"),
        State("inv-accommodates",     "value"),
        State("inv-bedrooms",         "value"),
        State("inv-bathrooms",        "value"),
        State("inv-beds",             "value"),
        State("inv-min-nights",       "value"),
        State("inv-availability-365", "value"),
        State("inv-instant-bookable", "value"),
        State("inv-superhost",        "value"),
        State("inv-host-listings",    "value"),
        State("inv-amenity-count",    "value"),
        State("inv-has_wifi",             "value"),
        State("inv-has_kitchen",          "value"),
        State("inv-has_washer",           "value"),
        State("inv-has_dryer",            "value"),
        State("inv-has_parking",          "value"),
        State("inv-has_air_conditioning", "value"),
        State("inv-has_heating",          "value"),
        State("inv-has_tv",               "value"),
        State("inv-has_self_check-in",    "value"),
        State("inv-has_coffee",           "value"),
        State("inv-has_hair_dryer",       "value"),
        State("inv-has_iron",             "value"),
        State("inv-has_gym",              "value"),
        State("inv-has_pool",             "value"),
        State("inv-has_hot_tub",          "value"),
        State("inv-has_elevator",         "value"),
        prevent_initial_call=True,
    )
    def run_prediction(
        n_clicks, city,
        neighbourhood, property_type,
        accommodates, bedrooms, bathrooms, beds,
        min_nights, availability_365,
        instant_bookable, superhost, host_listings, amenity_count,
        w_wifi, w_kitchen, w_washer, w_dryer, w_parking,
        w_ac, w_heating, w_tv, w_checkin, w_coffee,
        w_hairdryer, w_iron, w_gym, w_pool, w_hottub, w_elevator,
    ):
        if not n_clicks:
            from layouts.tab3_predictor import _empty_output_panel
            return _empty_output_panel()

        def flag(val):
            return 1 if val else 0

        form_values = {
            "neighbourhood":        neighbourhood or "Mission",
            "property_type":        property_type or "Entire rental unit",
            "accommodates":         accommodates or 2,
            "bathrooms_clean":      bathrooms or 1.0,
            "bedrooms":             bedrooms or 1,
            "beds":                 beds or 1,
            "minimum_nights":       min_nights or 2,
            "maximum_nights":       365,
            "availability_365":     availability_365 or 200,
            "instant_bookable_num": instant_bookable or 0,
            "host_is_superhost":    superhost or 0.0,
            "host_listings":        host_listings or 1,
            "amenity_count":        amenity_count or 20,
            "has_wifi":             flag(w_wifi),
            "has_kitchen":          flag(w_kitchen),
            "has_washer":           flag(w_washer),
            "has_dryer":            flag(w_dryer),
            "has_parking":          flag(w_parking),
            "has_air_conditioning": flag(w_ac),
            "has_heating":          flag(w_heating),
            "has_tv":               flag(w_tv),
            "has_self_check-in":    flag(w_checkin),
            "has_coffee":           flag(w_coffee),
            "has_hair_dryer":       flag(w_hairdryer),
            "has_iron":             flag(w_iron),
            "has_gym":              flag(w_gym),
            "has_pool":             flag(w_pool),
            "has_hot_tub":          flag(w_hottub),
            "has_elevator":         flag(w_elevator),
        }

        result = predict_price(form_values, city or CITIES[0])
        return build_output_panel(result)
