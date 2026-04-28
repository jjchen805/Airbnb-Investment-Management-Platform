"""
Tab 2 Callbacks — Superhost Advisor
=====================================
Handles:
  - City selection → swap listing data + superhost model
  - Listing dropdown selection → full advisor panel
  - dcc.Store sync: map click on Tab 1 auto-selects listing here
  - SHAP explanation via CatBoost native TreeSHAP
  - Recommendation engine: actionable, ranked by SHAP magnitude
  - AI agent: action plan + Q&A grounded in listing context
"""
import json
import joblib
import os
import numpy as np
import pandas as pd
from catboost import Pool

from dash import Input, Output, State, no_update, ctx
from layouts.tab2_advisor import (
    build_advisor_panel, _empty_panel, AMENITY_LABELS
)
from services.llm_agent import (
    build_advisor_agent_plan,
    build_advisor_agent_answer,
    agent_enabled,
)

# ── Load all city data + models at startup ────────────────────────────────────
CITIES = [
    c.replace("dashboard_listings_", "").replace(".csv", "")
    for c in os.listdir("data")
    if c.startswith("dashboard_listings_") and c.endswith(".csv")
]

ALL_DF = {}
for city in CITIES:
    df = pd.read_csv(f"data/dashboard_listings_{city}.csv")
    df["host_is_superhost"] = pd.to_numeric(df["host_is_superhost"], errors="coerce").fillna(0)
    ALL_DF[city] = df

ALL_SH_MODELS = {}
for city in CITIES:
    ALL_SH_MODELS[city] = {
        "model":    joblib.load(f"models/{city}/superhost_model.pkl"),
        "features": joblib.load(f"models/{city}/superhost_features.pkl"),
        "meta":     joblib.load(f"models/{city}/superhost_meta.pkl"),
    }

ALL_META = {}
for city in CITIES:
    with open(f"data/dashboard_meta_{city}.json") as f:
        ALL_META[city] = json.load(f)

print(f"  Advisor: loaded {len(CITIES)} cities — {CITIES}")


def _get_city_assets(city: str):
    city  = city if city in CITIES else CITIES[0]
    df    = ALL_DF[city]
    sh    = ALL_SH_MODELS[city]
    meta  = ALL_META[city]
    return df, sh["model"], sh["features"], sh["meta"]["fillna_medians"], meta


# ── Feature human labels ──────────────────────────────────────────────────────
FEATURE_LABELS = {
    "host_response_rate_clean":       "Response rate",
    "host_acceptance_rate_clean":     "Acceptance rate",
    "host_has_profile_pic_num":       "Profile picture",
    "host_identity_verified_num":     "Identity verified",
    "host_response_time_num":         "Response time",
    "host_tenure_years":              "Host tenure",
    "calculated_host_listings_count": "Listings count",
    "instant_bookable_num":           "Instant bookable",
    "number_of_reviews":              "Number of reviews",
    "review_scores_rating":           "Overall rating",
    "review_scores_cleanliness":      "Cleanliness score",
    "review_scores_communication":    "Communication score",
    "review_scores_checkin":          "Check-in score",
    "review_scores_location":         "Location score",
    "review_scores_value":            "Value score",
    "sentiment_polarity_mean":        "Review sentiment",
    "theme_cleanliness_mean":         "Cleanliness theme",
    "theme_communication_mean":       "Communication theme",
    "theme_checkin_mean":             "Check-in theme",
    "theme_location_mean":            "Location theme",
    "theme_amenities_mean":           "Amenities theme",
    "theme_value_mean":               "Value theme",
    "theme_comfort_mean":             "Comfort theme",
    "theme_accuracy_mean":            "Accuracy theme",
    "amenity_count":                  "Total amenities",
    "is_entire_home":                 "Entire home",
    "is_private_room":                "Private room",
    **{col: AMENITY_LABELS[col] for col in AMENITY_LABELS},
}

# ── Recommendation templates ──────────────────────────────────────────────────
RECOMMENDATION_TEMPLATES = {
    "host_has_profile_pic_num": {
        "title":  "Add a host profile picture",
        "reason": "Superhosts almost universally have profile pictures. It builds trust "
                  "with guests before they book, and the model treats this as a strong signal.",
        "impact": "high",
    },
    "host_identity_verified_num": {
        "title":  "Complete identity verification",
        "reason": "Verified hosts are significantly more likely to achieve Superhost status. "
                  "Guests feel safer booking with verified hosts.",
        "impact": "high",
    },
    "host_response_rate_clean": {
        "title":  "Improve your response rate",
        "reason": "Airbnb requires a 90%+ response rate for Superhost status. "
                  "Your current rate is pulling down your probability score.",
        "impact": "high",
    },
    "host_response_time_num": {
        "title":  "Respond faster to enquiries",
        "reason": "Superhosts typically respond within an hour. Faster responses improve "
                  "your response time score, a direct Superhost eligibility criterion.",
        "impact": "high",
    },
    "host_acceptance_rate_clean": {
        "title":  "Increase your acceptance rate",
        "reason": "A higher acceptance rate is strongly associated with Superhost status. "
                  "Aim for 90%+ to meet Airbnb's eligibility requirements.",
        "impact": "high",
    },
    "instant_bookable_num": {
        "title":  "Enable instant booking",
        "reason": "Instant bookable listings have higher acceptance rates by design, "
                  "and the model associates this with stronger Superhost likelihood.",
        "impact": "medium",
    },
    "has_self_check-in": {
        "title":  "Add self check-in",
        "reason": "Self check-in is associated with better check-in review scores and "
                  "gives guests more flexibility — both linked to Superhost performance.",
        "impact": "medium",
    },
    "has_kitchen": {
        "title":  "Add a kitchen or kitchenette",
        "reason": "Kitchen access is one of the most valued amenities. "
                  "Listings with kitchens consistently earn higher amenity review scores.",
        "impact": "medium",
    },
    "has_parking": {
        "title":  "Offer parking",
        "reason": "Parking is a premium amenity associated with higher overall "
                  "satisfaction scores, especially in dense urban markets.",
        "impact": "medium",
    },
    "has_air_conditioning": {
        "title":  "Add air conditioning",
        "reason": "AC is increasingly expected by guests and is associated with "
                  "higher comfort scores in review themes.",
        "impact": "medium",
    },
    "has_tv": {
        "title":  "Add a TV",
        "reason": "TV is a standard expected amenity. Its absence can negatively "
                  "affect amenity satisfaction review scores.",
        "impact": "low",
    },
    "has_washer": {
        "title":  "Add a washer",
        "reason": "A washer is especially valued for longer stays and is associated "
                  "with higher amenity scores.",
        "impact": "low",
    },
    "has_hair_dryer": {
        "title":  "Provide a hair dryer",
        "reason": "A small addition that guests frequently mention positively in reviews. "
                  "It contributes to higher amenity satisfaction scores.",
        "impact": "low",
    },
    "amenity_count": {
        "title":  "Increase your total amenity count",
        "reason": "Superhosts typically offer 35+ amenities. More amenities correlate "
                  "with higher guest satisfaction scores overall.",
        "impact": "medium",
    },
}


# ── Core logic ────────────────────────────────────────────────────────────────

def _compute_shap(row, model, features, fillna_medians):
    X = pd.DataFrame([{f: row.get(f, fillna_medians.get(f, 0)) for f in features}])
    X = X.fillna(fillna_medians)
    shap_matrix = model.get_feature_importance(data=Pool(X), type="ShapValues")
    return dict(zip(features, shap_matrix[0, :-1])), float(shap_matrix[0, -1])


def _build_shap_drivers(shap_vals, top_n=10):
    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    drivers, seen = [], set()
    for feat, val in sorted_feats:
        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
        if label in seen:
            continue
        seen.add(label)
        drivers.append({
            "label":     label,
            "shap_val":  float(val),
            "direction": "up" if val > 0 else "down",
        })
        if len(drivers) >= top_n:
            break
    return drivers


def _build_recommendations(row, shap_vals, fillna_medians):
    candidates = []
    for feat, template in RECOMMENDATION_TEMPLATES.items():
        if feat not in shap_vals:
            continue
        val      = row.get(feat, fillna_medians.get(feat, 0))
        shap_val = shap_vals.get(feat, 0)
        is_weak  = False
        if feat == "host_response_rate_clean":
            is_weak = (val is None or val != val or float(val) < 90)
        elif feat == "host_acceptance_rate_clean":
            is_weak = (val is None or val != val or float(val) < 90)
        elif feat == "host_response_time_num":
            is_weak = (val is None or val != val or float(val) > 1)
        elif feat == "amenity_count":
            is_weak = (val is None or val != val or float(val) < 30)
        elif feat in {
            "host_has_profile_pic_num", "host_identity_verified_num",
            "instant_bookable_num", "has_self_check-in", "has_kitchen",
            "has_parking", "has_air_conditioning", "has_tv",
            "has_washer", "has_hair_dryer",
        }:
            is_weak = (val == 0 or val is None or val != val)
        else:
            is_weak = shap_val < 0

        if is_weak:
            candidates.append({
                "title":          template["title"],
                "reason":         template["reason"],
                "impact":         template["impact"],
                "shap_magnitude": abs(shap_val),
            })

    impact_order = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda x: (impact_order.get(x["impact"], 9), -x["shap_magnitude"]))
    return [{"title": c["title"], "reason": c["reason"], "impact": c["impact"]}
            for c in candidates[:5]]


def _build_strengths_weaknesses(row, shap_vals):
    strengths, weaknesses = [], []
    checks = {
        "host_has_profile_pic_num":   ("Profile picture set",       "No profile picture"),
        "host_identity_verified_num": ("Identity verified",          "Identity not verified"),
        "instant_bookable_num":       ("Instant booking enabled",    "Instant booking not enabled"),
        "has_self_check-in":          ("Self check-in available",    "No self check-in"),
        "has_kitchen":                ("Kitchen provided",           "No kitchen"),
        "has_parking":                ("Parking available",          "No parking"),
        "has_air_conditioning":       ("Air conditioning available", "No air conditioning"),
        "has_tv":                     ("TV provided",                "No TV"),
        "has_washer":                 ("Washer available",           "No washer"),
    }
    for feat, (pos_label, neg_label) in checks.items():
        val = row.get(feat, 0)
        if val and val == val and float(val) > 0:
            strengths.append(pos_label)
        else:
            weaknesses.append(neg_label)

    rr = row.get("host_response_rate_clean")
    if rr and rr == rr:
        if float(rr) >= 90:
            strengths.append(f"Strong response rate ({rr:.0f}%)")
        else:
            weaknesses.append(f"Low response rate ({rr:.0f}%)")
    else:
        weaknesses.append("Response rate not set")

    sent = row.get("sentiment_polarity_mean")
    if sent and sent == sent:
        if float(sent) > 0.45:
            strengths.append(f"High review sentiment ({sent:.2f})")
        elif float(sent) < 0.30:
            weaknesses.append(f"Low review sentiment ({sent:.2f})")

    theme_feats = [f for f in shap_vals if f.startswith("theme_") and "_mean" in f]
    if theme_feats:
        best  = max(theme_feats, key=lambda f: shap_vals[f])
        worst = min(theme_feats, key=lambda f: shap_vals[f])
        if shap_vals[best] > 0:
            strengths.append(f"Strong {FEATURE_LABELS.get(best, best).lower()}")
        if shap_vals[worst] < -0.05:
            weaknesses.append(f"Weak {FEATURE_LABELS.get(worst, worst).lower()}")

    ac = row.get("amenity_count", 0)
    if ac >= 35:
        strengths.append(f"High amenity count ({int(ac)} amenities)")
    elif ac < 20:
        weaknesses.append(f"Low amenity count ({int(ac)} amenities)")

    return strengths[:5], weaknesses[:5]


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_advisor_callbacks(app):

    # ── Neighbourhood filter OR map click → update listing dropdown ───────
    # Merged into one callback to avoid duplicate Output on adv-listing-select.value
    @app.callback(
        Output("adv-listing-select",  "options"),
        Output("adv-listing-select",  "value"),
        Output("adv-listing-count",   "children"),
        Input("adv-neighbourhood-filter", "value"),
        Input("selected-listing-id",      "data"),
        State("main-tabs",                "value"),
        State("selected-city",            "data"),
        State("adv-listing-select",       "value"),
    )
    def update_listing_dropdown(neighbourhood, map_listing_id, current_tab, city, current_value):
        from layouts.tab2_advisor import _build_listing_options
        trigger  = ctx.triggered_id
        df       = ALL_DF.get(city, next(iter(ALL_DF.values())))
        options  = _build_listing_options(df, neighbourhood)

        # Map click on Tab 1 → jump straight to that listing, ignore filter
        if trigger == "selected-listing-id":
            if current_tab == "tab-advisor" and map_listing_id and map_listing_id in df["id"].values:
                new_value = map_listing_id
            else:
                new_value = current_value
        else:
            # Neighbourhood filter changed — keep selection only if still valid
            valid_values = {o["value"] for o in options}
            new_value    = current_value if current_value in valid_values else None

        nbhd_label = neighbourhood if neighbourhood and neighbourhood != "__all__" else "all neighbourhoods"
        count_text = f"{len(options):,} listings in {nbhd_label}"

        return options, new_value, count_text

    # ── City change → reset neighbourhood filter ───────────────────────────
    @app.callback(
        Output("adv-neighbourhood-filter", "options"),
        Output("adv-neighbourhood-filter", "value"),
        Input("selected-city", "data"),
    )
    def reset_filters_on_city_change(city):
        df             = ALL_DF.get(city, next(iter(ALL_DF.values())))
        neighbourhoods = sorted(df["neighbourhood_top"].dropna().unique().tolist())
        options        = [{"label": "All neighbourhoods", "value": "__all__"}] + [
            {"label": n, "value": n} for n in neighbourhoods
        ]
        return options, "__all__"

    # ── Listing selected → full advisor panel ──────────────────────────────
    @app.callback(
        Output("adv-output-panel",  "children"),
        Output("adv-agent-context", "data"),
        Input("adv-listing-select", "value"),
        State("selected-city",      "data"),
    )
    def update_advisor_panel(listing_id, city):
        empty_ctx = {"city": city or CITIES[0], "has_listing": False}
        if listing_id is None:
            return _empty_panel(), empty_ctx

        try:
            listing_id = int(listing_id)
        except (ValueError, TypeError):
            return _empty_panel(), empty_ctx

        df, model, features, fillna_medians, meta = _get_city_assets(city)
        matches = df[df["id"] == listing_id]
        if matches.empty:
            return _empty_panel(), empty_ctx

        row = matches.iloc[0].to_dict()

        try:
            shap_vals, _ = _compute_shap(row, model, features, fillna_medians)
        except Exception:
            shap_vals = {}

        shap_drivers          = _build_shap_drivers(shap_vals, top_n=10)
        recommendations       = _build_recommendations(row, shap_vals, fillna_medians)
        strengths, weaknesses = _build_strengths_weaknesses(row, shap_vals)

        agent_context = {
            "city":            city or CITIES[0],
            "has_listing":     True,
            "row":             row,
            "shap_drivers":    shap_drivers,
            "recommendations": recommendations,
            "strengths":       strengths,
            "weaknesses":      weaknesses,
        }

        return build_advisor_panel(
            row=row,
            shap_drivers=shap_drivers,
            recommendations=recommendations,
            strengths=strengths,
            weaknesses=weaknesses,
        ), agent_context

    # ── Toggle AI controls based on whether a listing is selected ─────────
    @app.callback(
        Output("adv-agent-question",    "disabled"),
        Output("adv-agent-ask-btn",     "disabled"),
        Output("adv-agent-generate-btn","disabled"),
        Output("adv-agent-status",      "children"),
        Input("adv-agent-context",      "data"),
    )
    def toggle_advisor_agent_controls(agent_context):
        has_listing = bool(agent_context and agent_context.get("has_listing"))
        if not agent_enabled():
            return True, True, True, "AI agent disabled — set ENABLE_AGENT=true in .env"
        if has_listing:
            return False, False, False, "Listing selected — ask a question or generate an action plan."
        return True, True, True, "Select a listing to unlock AI features."

    # ── AI: Q&A or action plan ─────────────────────────────────────────────
    @app.callback(
        Output("adv-agent-content",     "children"),
        Input("adv-agent-ask-btn",      "n_clicks"),
        Input("adv-agent-generate-btn", "n_clicks"),
        State("adv-agent-question",     "value"),
        State("adv-agent-context",      "data"),
        prevent_initial_call=True,
    )
    def generate_advisor_agent_output(ask_clicks, plan_clicks, question, agent_context):
        trigger = ctx.triggered_id
        if not agent_context or not agent_context.get("has_listing"):
            return "Please select a listing first."

        unavailable = (
            "AI agent unavailable. Add to .env:\n\n"
            "```\nENABLE_AGENT=true\n"
            "AGENT_PROVIDER=anthropic\n"
            "ANTHROPIC_API_KEY=sk-ant-...\n```"
        )

        if trigger == "adv-agent-ask-btn":
            q = (question or "").strip()
            if not q:
                return "Please enter a question."
            listing_context = {
                "row":             agent_context.get("row", {}),
                "shap_drivers":    agent_context.get("shap_drivers", []),
                "recommendations": agent_context.get("recommendations", []),
                "strengths":       agent_context.get("strengths", []),
                "weaknesses":      agent_context.get("weaknesses", []),
            }
            text = build_advisor_agent_answer(
                city=agent_context.get("city", CITIES[0]),
                question=q,
                listing_context=listing_context,
            )
            return text or unavailable

        # Generate action plan
        text = build_advisor_agent_plan(
            city=agent_context.get("city", CITIES[0]),
            row=agent_context.get("row", {}),
            shap_drivers=agent_context.get("shap_drivers", []),
            recommendations=agent_context.get("recommendations", []),
            strengths=agent_context.get("strengths", []),
            weaknesses=agent_context.get("weaknesses", []),
        )
        return text or unavailable
