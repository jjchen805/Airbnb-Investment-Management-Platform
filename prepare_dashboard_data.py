"""
Phase 1 — Dashboard Data Preparation
=====================================
Run this ONCE after training your models.
Produces:
  data/dashboard_listings.csv   — all listings enriched with model predictions
  data/dashboard_meta.json      — neighbourhood stats, amenity lifts, model info

Usage:
  python prepare_dashboard_data.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--city", required=True)
args = parser.parse_args()

DATA_PATH = f"data/clean_airbnb_dataset_{args.city}.csv"
MODEL_DIR = f"models/{args.city}"
OUT_DATA  = f"data/dashboard_listings_{args.city}.csv"
OUT_META  = f"data/dashboard_meta_{args.city}.json"

AMENITY_COLS = [
    "has_wifi", "has_kitchen", "has_washer", "has_dryer", "has_parking",
    "has_air_conditioning", "has_heating", "has_tv", "has_self_check-in",
    "has_coffee", "has_hair_dryer", "has_iron", "has_gym", "has_pool",
    "has_hot_tub", "has_elevator",
]

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

# Clean price column — keep nulls, just parse
df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")

# Cap outliers only for benchmark stats (not for filtering rows)
price_cap_value = float(df["price_clean"].quantile(0.99))
print(f"  Price 99th pct cap: ${price_cap_value:.0f} (used for stats only, not row filtering)")

# ── 2. Load model artifacts ───────────────────────────────────────────────────
print("Loading model artifacts...")

price_model      = joblib.load(f"{MODEL_DIR}/price_model.pkl")
price_features   = joblib.load(f"{MODEL_DIR}/price_features.pkl")
price_meta       = joblib.load(f"{MODEL_DIR}/price_meta.pkl")
print(f"  Price model loaded ({len(price_features)} features)")

superhost_model    = joblib.load(f"{MODEL_DIR}/superhost_model.pkl")
superhost_features = joblib.load(f"{MODEL_DIR}/superhost_features.pkl")
superhost_meta     = joblib.load(f"{MODEL_DIR}/superhost_meta.pkl")
print(f"  Superhost model loaded ({len(superhost_features)} features, AUC={superhost_meta['auc_score']})")

# ── 3. Predicted prices (all rows) ───────────────────────────────────────────
print("Generating predicted prices for all listings...")

group_medians  = price_meta["group_medians"]
fillna_medians = price_meta["fillna_medians"]

X_price = df.copy()
for col in ["neighbourhood_top", "property_type_simple"]:
    X_price[f"{col}_median_price"] = X_price[col].map(group_medians[col])

X_price = pd.get_dummies(X_price, columns=["neighbourhood_top", "property_type_simple"], drop_first=True)
X_price = X_price.reindex(columns=price_features, fill_value=0)
X_price = X_price.fillna(fillna_medians)

df["predicted_price"] = np.expm1(price_model.predict(X_price)).round(0).astype(int)
print(f"  Predicted price range: ${df['predicted_price'].min()} – ${df['predicted_price'].max()}")

# ── 4. Superhost probabilities (all rows) ─────────────────────────────────────
print("Generating superhost probabilities for all listings...")

sh_fillna = superhost_meta["fillna_medians"]
X_sh = df[superhost_features].fillna(sh_fillna)
df["superhost_probability"] = superhost_model.predict_proba(X_sh)[:, 1].round(4)

thresholds = superhost_meta["prob_thresholds"]
print(f"  Prob range: {df['superhost_probability'].min():.3f} – {df['superhost_probability'].max():.3f}")
print(f"  High potential (>{thresholds['high']}): {(df['superhost_probability'] > thresholds['high']).sum():,} listings")
print(f"  Low potential  (<{thresholds['low']}):  {(df['superhost_probability'] < thresholds['low']).sum():,} listings")

# ── 5. Select columns for dashboard ──────────────────────────────────────────
print("Selecting dashboard columns...")

dashboard_cols = [
    # Identity
    "id", "name", "listing_url",
    # Location
    "latitude", "longitude", "neighbourhood_top", "neighbourhood_cleansed",
    # Property
    "property_type", "property_type_simple", "room_type",
    "accommodates", "bedrooms", "bathrooms_clean", "beds",
    # Host
    "host_name", "host_is_superhost", "host_response_rate_clean",
    "host_acceptance_rate_clean", "host_has_profile_pic_num",
    "host_identity_verified_num", "host_tenure_years",
    "host_response_time_num", "calculated_host_listings_count",
    "instant_bookable_num",
    # Pricing
    "price_clean", "predicted_price",
    # Superhost probability
    "superhost_probability",
    # Reviews
    "number_of_reviews", "review_scores_rating",
    "review_scores_cleanliness", "review_scores_communication",
    "review_scores_checkin", "review_scores_location", "review_scores_value",
    # NLP / sentiment
    "sentiment_polarity_mean", "review_count_from_text",
    "theme_cleanliness_mean", "theme_communication_mean",
    "theme_checkin_mean", "theme_location_mean",
    "theme_amenities_mean", "theme_value_mean", "theme_comfort_mean",
    "theme_accuracy_mean",
    "theme_cleanliness_positive_mean", "theme_cleanliness_negative_mean",
    "theme_communication_positive_mean", "theme_communication_negative_mean",
    "theme_checkin_positive_mean", "theme_checkin_negative_mean",
    "theme_location_positive_mean", "theme_location_negative_mean",
    "theme_amenities_positive_mean", "theme_amenities_negative_mean",
    "theme_value_positive_mean", "theme_value_negative_mean",
    "theme_comfort_positive_mean", "theme_comfort_negative_mean",
    "theme_accuracy_positive_mean", "theme_accuracy_negative_mean",
    # Amenities
    "amenity_count", "has_wifi", "has_kitchen", "has_washer", "has_dryer",
    "has_parking", "has_air_conditioning", "has_heating", "has_tv",
    "has_self_check-in", "has_gym", "has_pool", "has_hot_tub",
    "has_elevator", "has_iron", "has_coffee", "has_hair_dryer",
    # Flags
    "is_entire_home", "is_private_room", "is_shared_room",
    # Availability
    "availability_60", "availability_365",
    "minimum_nights", "maximum_nights",
]

dashboard_cols = [c for c in dashboard_cols if c in df.columns]
df_dash = df[dashboard_cols].copy()
df_dash.to_csv(OUT_DATA, index=False)
print(f"  Saved: {OUT_DATA} ({len(df_dash):,} rows, {len(df_dash.columns)} columns)")

# ── 6. Reference stats (using capped prices for fair benchmarks) ──────────────
print("Computing reference stats...")

df_capped = df[df["price_clean"].notna() & (df["price_clean"] <= price_cap_value)].copy()

nbhd_stats = df_capped.groupby("neighbourhood_top")["price_clean"].agg(
    median_price="median", mean_price="mean", listing_count="count"
).round(1).reset_index()

nbhd_coords = df.groupby("neighbourhood_top")[["latitude", "longitude"]].median().round(5).reset_index()
nbhd_stats  = nbhd_stats.merge(nbhd_coords, on="neighbourhood_top")

prop_stats = df_capped.groupby("property_type_simple")["price_clean"].agg(
    median_price="median", mean_price="mean", listing_count="count"
).round(1).reset_index()

investor_num_defaults = {
    k: round(float(v), 2)
    for k, v in fillna_medians.items()
    if k in [
        "accommodates", "bathrooms_clean", "bedrooms", "beds",
        "minimum_nights", "maximum_nights", "availability_60",
        "availability_365", "calculated_host_listings_count", "amenity_count"
    ]
}

# ── 7. Amenity price lifts ────────────────────────────────────────────────────
print("Computing amenity lifts...")

amenity_lifts = {}
for col in AMENITY_COLS:
    if col not in df_capped.columns:
        continue
    with_g    = df_capped[df_capped[col] == 1]["price_clean"]
    without_g = df_capped[df_capped[col] == 0]["price_clean"]
    if len(with_g) >= 10 and len(without_g) >= 10:
        amenity_lifts[col] = round(float(with_g.median() - without_g.median()), 1)

amenity_lifts_by_nbhd = {}
for nbhd in df_capped["neighbourhood_top"].unique():
    sub = df_capped[df_capped["neighbourhood_top"] == nbhd]
    amenity_lifts_by_nbhd[nbhd] = {}
    for col in AMENITY_COLS:
        if col not in df_capped.columns:
            continue
        w  = sub[sub[col] == 1]["price_clean"]
        wo = sub[sub[col] == 0]["price_clean"]
        amenity_lifts_by_nbhd[nbhd][col] = (
            round(float(w.median() - wo.median()), 1)
            if len(w) >= 5 and len(wo) >= 5
            else amenity_lifts.get(col)
        )

print(f"  Computed lifts for {len(amenity_lifts)} amenities across {len(amenity_lifts_by_nbhd)} neighbourhoods")

# ── 8. Write dashboard_meta.json ──────────────────────────────────────────────
sh_col = pd.to_numeric(df["host_is_superhost"], errors="coerce")

meta = {
    "price_cap_value":      price_cap_value,
    "overall_median_price": float(df_capped["price_clean"].median()),
    "overall_mean_price":   float(df_capped["price_clean"].mean().round(1)),
    "total_listings":       len(df),
    "superhost_pct":        round(float(sh_col.mean()) * 100, 1),
    "neighbourhoods":       nbhd_stats.to_dict(orient="records"),
    "property_types":       prop_stats.to_dict(orient="records"),
    "investor_num_defaults": investor_num_defaults,
    "group_medians":        group_medians,
    "fillna_medians":       {k: float(v) if pd.notna(v) else 0.0 for k, v in fillna_medians.items()},
    "amenity_lifts":        amenity_lifts,
    "amenity_lifts_by_nbhd": amenity_lifts_by_nbhd,
    "superhost_model_auc":  superhost_meta["auc_score"],
    "superhost_thresholds": superhost_meta["prob_thresholds"],
    "superhost_actionable_features": superhost_meta["actionable_features"],
}

with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"  Saved: {OUT_META}")
print("\n✓ Data prep complete.")
print(f"  dashboard_listings.csv : {len(df_dash):,} listings (all rows, no price filter)")
print(f"  Neighbourhoods         : {df['neighbourhood_top'].nunique()}")
print(f"  Predicted price median : ${df['predicted_price'].median():.0f}")
print(f"  Superhost prob median  : {df['superhost_probability'].median():.2f}")
print(f"  Listings with null price (shown in dashboard): {df['price_clean'].isna().sum():,}")
