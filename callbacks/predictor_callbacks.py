"""
Tab 3 Callbacks — Investor Price Predictor
==========================================
Uses CatBoost native TreeSHAP for local explanations.
Includes what-if scenario engine and AI investment brief/Q&A.
"""
import json
import joblib
import os
import numpy as np
import pandas as pd
from catboost import Pool

from dash import Input, Output, State, no_update, ctx
from layouts.tab3_predictor import build_output_panel
from services.llm_agent import (
    build_predictor_whatif_plan,
    build_investor_agent_answer,
    build_investor_agent_brief,
    agent_enabled,
)

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

# Estimated one-time costs ($) for adding each amenity (for what-if ROI ranking)
AMENITY_COSTS = {
    "has_wifi":            35,
    "has_kitchen":         1200,
    "has_washer":          700,
    "has_dryer":           650,
    "has_parking":         200,
    "has_air_conditioning":1800,
    "has_heating":         600,
    "has_tv":              350,
    "has_self_check-in":   120,
    "has_coffee":          80,
    "has_hair_dryer":      40,
    "has_iron":            35,
    "has_gym":             3000,
    "has_pool":            8000,
    "has_hot_tub":         4500,
    "has_elevator":        6000,
}

HOST_SETUP_COSTS = {
    "instant_bookable_num": 0,
    "host_is_superhost":    350,
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


def _build_feature_row(form_values, price_features, group_medians,
                       fillna_medians, nbhd_coords):
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


def _compute_shap(X_df, price_model, price_features, predicted_price):
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


def _get_amenity_lift(amenity_col, neighbourhood, meta):
    nbhd_data = meta.get("amenity_lifts_by_nbhd", {}).get(neighbourhood, {})
    val = nbhd_data.get(amenity_col)
    if val is not None:
        return float(val)
    return float(meta.get("amenity_lifts", {}).get(amenity_col, 0))


def predict_price(form_values: dict, city: str) -> dict:
    price_model, price_features, price_meta_city, meta = _get_city_assets(city)

    group_medians  = price_meta_city["group_medians"]
    fillna_medians = price_meta_city["fillna_medians"]
    nbhd_coords    = {r["neighbourhood_top"]: (r["latitude"], r["longitude"])
                      for r in meta["neighbourhoods"]}

    X_df            = _build_feature_row(form_values, price_features,
                                         group_medians, fillna_medians, nbhd_coords)
    y_log_pred      = price_model.predict(X_df)[0]
    predicted_price = float(np.expm1(y_log_pred))
    drivers         = _compute_shap(X_df, price_model, price_features, predicted_price)

    neighbourhood = form_values["neighbourhood"]
    unchecked     = [col for col in AMENITY_COLS if not form_values.get(col, 0)]
    amenity_gaps  = sorted(
        [{"col": col, "label": AMENITY_LABELS[col],
          "market_lift": _get_amenity_lift(col, neighbourhood, meta)}
         for col in unchecked
         if _get_amenity_lift(col, neighbourhood, meta) > 10],
        key=lambda x: x["market_lift"], reverse=True
    )[:4]

    nbhd_medians = {r["neighbourhood_top"]: r["median_price"] for r in meta["neighbourhoods"]}
    prop_medians = {r["property_type_simple"]: r["median_price"] for r in meta["property_types"]}
    nbhd_median  = nbhd_medians.get(neighbourhood, meta["overall_median_price"])
    prop_median  = prop_medians.get(form_values["property_type"], meta["overall_median_price"])
    pct_vs_nbhd  = ((predicted_price - nbhd_median) / nbhd_median) * 100

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


# ── What-if scenario engine ───────────────────────────────────────────────────

def _booking_win_rate_proxy(predicted_price: float, nbhd_median: float) -> float:
    if nbhd_median <= 0:
        return 50.0
    ratio = predicted_price / nbhd_median
    score = 100.0 - abs(ratio - 1.0) * 120.0
    return float(max(0.0, min(100.0, score)))


def _build_candidate_mods(form_values, city, budget, meta):
    neighbourhood    = form_values["neighbourhood"]
    missing_amenities = [col for col in AMENITY_COLS if not form_values.get(col, 0)]

    amenity_rank = sorted(
        [(col, _get_amenity_lift(col, neighbourhood, meta),
          float(AMENITY_COSTS.get(col, 300)))
         for col in missing_amenities
         if _get_amenity_lift(col, neighbourhood, meta) > 0],
        key=lambda x: x[1] / max(x[2], 1.0), reverse=True
    )

    candidates, seen = [], set()

    def add(name, updates, cost, changes):
        if cost > budget:
            return
        key = (name, tuple(sorted(updates.items())))
        if key in seen:
            return
        seen.add(key)
        candidates.append({"name": name, "updates": updates,
                           "estimated_budget": float(cost), "changes": changes})

    # Single amenity upgrades
    for col, _, cost in amenity_rank[:8]:
        add(f"Add {AMENITY_LABELS[col]}", {col: 1}, cost,
            [f"Enable {AMENITY_LABELS[col]}"])

    # Amenity bundles (top ROI pairs)
    top = amenity_rank[:6]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c1, _, cost1 = top[i]
            c2, _, cost2 = top[j]
            add(f"Bundle: {AMENITY_LABELS[c1]} + {AMENITY_LABELS[c2]}",
                {c1: 1, c2: 1}, cost1 + cost2,
                [f"Add {AMENITY_LABELS[c1]}", f"Add {AMENITY_LABELS[c2]}"])

    # Minimum nights what-if
    current_min = int(form_values.get("minimum_nights", 2))
    for mn in [1, 2, 3, 5, 7]:
        if mn != current_min:
            add(f"Min nights = {mn}", {"minimum_nights": mn}, 0,
                [f"Set minimum nights to {mn}"])

    # Host setup
    if not form_values.get("instant_bookable_num", 0):
        add("Enable instant book", {"instant_bookable_num": 1},
            HOST_SETUP_COSTS["instant_bookable_num"], ["Enable instant book"])
    if not form_values.get("host_is_superhost", 0):
        add("Superhost readiness", {"host_is_superhost": 1.0},
            HOST_SETUP_COSTS["host_is_superhost"], ["Target Superhost standards"])

    return candidates


def _apply_mods_to_form(form_values, updates):
    modified = dict(form_values)
    added = [k for k, v in updates.items()
             if k in AMENITY_COLS and int(v) == 1 and not modified.get(k, 0)]
    modified.update(updates)
    if added:
        modified["amenity_count"] = int(modified.get("amenity_count", 0)) + len(added)
    return modified


def generate_whatif_plan(form_values, city, budget, base_result):
    city = city or CITIES[0]
    price_model, price_features, price_meta_city, meta = _get_city_assets(city)

    candidates  = _build_candidate_mods(form_values, city, budget, meta)
    base_price  = float(base_result["predicted_price"])
    base_win    = _booking_win_rate_proxy(base_price, float(base_result["nbhd_median"]))

    ranked = []
    for c in candidates:
        modified = _apply_mods_to_form(form_values, c["updates"])
        sim      = predict_price(modified, city)
        pred     = float(sim["predicted_price"])
        uplift   = pred - base_price
        win_new  = _booking_win_rate_proxy(pred, float(sim["nbhd_median"]))
        ranked.append({
            "name":             c["name"],
            "estimated_budget": round(c["estimated_budget"], 2),
            "predicted_price":  pred,
            "uplift_usd":       uplift,
            "uplift_pct":       (uplift / base_price * 100.0) if base_price > 0 else 0.0,
            "win_rate_delta":   win_new - base_win,
            "changes":          c["changes"],
            "objective":        uplift + (win_new - base_win) * 0.8,
        })

    ranked.sort(key=lambda x: x["objective"], reverse=True)
    return {
        "city":       city,
        "budget":     float(budget),
        "base_price": base_price,
        "scenarios":  ranked[:3],
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_predictor_callbacks(app):

    # ── Main prediction ────────────────────────────────────────────────────
    @app.callback(
        Output("inv-output-panel",   "children"),
        Output("inv-whatif-context", "data"),
        Output("inv-agent-context",  "data"),
        Input("inv-predict-btn",     "n_clicks"),
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
            return _empty_output_panel(), None, None

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

        city   = city or CITIES[0]
        result = predict_price(form_values, city)
        agent_context = {
            "city":        city,
            "form_values": form_values,
            "result":      result,
            "budget":      1500.0,
        }
        return build_output_panel(result, {}), None, agent_context

    # ── AI: brief, Q&A, or what-if plan ───────────────────────────────────
    @app.callback(
        Output("inv-agent-response",  "children"),
        Output("inv-output-panel",    "children", allow_duplicate=True),
        Output("inv-whatif-context",  "data",     allow_duplicate=True),
        Input("inv-agent-ask-btn",    "n_clicks"),
        Input("inv-agent-brief-btn",  "n_clicks"),
        Input("inv-whatif-generate-btn","n_clicks"),
        State("inv-agent-question",   "value"),
        State("inv-whatif-budget",    "value"),
        State("inv-whatif-context",   "data"),
        State("inv-agent-context",    "data"),
        prevent_initial_call=True,
    )
    def generate_investor_agent_output(
        ask_clicks, brief_clicks, whatif_clicks,
        question, budget_input, whatif_context, agent_context
    ):
        trigger = ctx.triggered_id
        if not agent_context:
            return "Run a prediction first.", no_update, no_update

        if brief_clicks == 0 and ask_clicks == 0 and whatif_clicks == 0:
            return no_update, no_update, no_update
    
        city        = agent_context.get("city", CITIES[0])
        form_values = agent_context.get("form_values", {})
        result      = agent_context.get("result", {})

        unavailable = (
            "AI agent unavailable. Add to .env:\n\n"
            "```\nENABLE_AGENT=true\n"
            "AGENT_PROVIDER=anthropic\n"
            "ANTHROPIC_API_KEY=sk-ant-...\n```"
        )

        if trigger == "inv-agent-ask-btn":
            q = (question or "").strip()
            if not q:
                return "Please enter a question.", no_update, no_update
            text = build_investor_agent_answer(
                city=city,
                question=q,
                investment_context={
                    "result":         result,
                    "whatif_top3":    (whatif_context or {}).get("scenarios", []),
                    "budget":         float(budget_input) if budget_input else 1500.0,
                },
            )
            return (text or unavailable), no_update, no_update

        if trigger == "inv-agent-brief-btn":
            text = build_investor_agent_brief(
                city=city, form_values=form_values, result=result
            )
            return (text or unavailable), no_update, no_update
        
        if trigger == "inv-whatif-generate-btn":
            budget = float(budget_input) if budget_input and float(budget_input) > 0 else 1500.0
            whatif = generate_whatif_plan(form_values, city, budget, result)
            text   = build_predictor_whatif_plan(
                city=whatif.get("city", city),
                budget=budget,
                base_price=float(whatif.get("base_price", 0.0)),
                scenarios=whatif.get("scenarios", []),
            )

            if text:
                whatif["llm_narrative"] = text
                
            return (text or unavailable), build_output_panel(result, whatif), whatif
        
            
        
        return no_update, no_update, no_update  # safety fallback