import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--city", required=True, help="e.g. sf, nyc, chicago")
parser.add_argument("--data", required=True, help="Path to clean_airbnb_dataset_{city}.csv")
args = parser.parse_args()
# ── 0. Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = f"models/{args.city}"
DATA_PATH = args.data
os.makedirs(MODEL_DIR, exist_ok=True)
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# ── 2. Target ─────────────────────────────────────────────────────────────────
y = pd.to_numeric(df["host_is_superhost"], errors="coerce")
valid_idx = y.notna()
df = df[valid_idx].copy()
y = y[valid_idx].astype(int)

print(f"Total rows: {len(df):,}")
print(f"Superhost rate: {y.mean()*100:.1f}%")

# ── 3. Feature list ────────────────────────────────────────────────────────────
# Split into two groups for transparency:
#   - host_setup: things a host can directly control (used for recommendations)
#   - review_based: earned through guest experience (surface when available)
#
# Note: calculated_host_listings_count is negative correlated (-0.26) —
# hosts with many listings are less likely to be superhosts (property managers).
# host_response_time_num: 1=within an hour, 4=a few days (lower = better).

superhost_features = [
    # Host setup — actionable
    "host_response_rate_clean",
    "host_acceptance_rate_clean",
    "host_has_profile_pic_num",
    "host_identity_verified_num",
    "host_response_time_num",
    "host_tenure_years",
    "calculated_host_listings_count",
    "instant_bookable_num",

    # Engagement signal
    "number_of_reviews",

    # Review scores — earned outcomes
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_communication",
    "review_scores_checkin",
    "review_scores_location",
    "review_scores_value",

    # NLP sentiment & themes — review-based
    "sentiment_polarity_mean",
    "theme_cleanliness_mean",
    "theme_communication_mean",
    "theme_checkin_mean",
    "theme_location_mean",
    "theme_amenities_mean",
    "theme_value_mean",
    "theme_comfort_mean",
    "theme_accuracy_mean",

    # Amenities — actionable
    "amenity_count",
    "has_wifi",
    "has_kitchen",
    "has_self_check-in",
    "has_parking",
    "has_air_conditioning",
    "has_heating",
    "has_tv",
    "has_washer",
    "has_dryer",
    "has_iron",
    "has_hair_dryer",
    "has_coffee",
    "has_pool",
    "has_gym",
    "has_hot_tub",
    "has_elevator",

    # Property type
    "is_entire_home",
    "is_private_room",
    "is_shared_room",
]

superhost_features = [f for f in superhost_features if f in df.columns]
print(f"\nFeatures used: {len(superhost_features)}")

# ── 4. Train / test split ──────────────────────────────────────────────────────
X = df[superhost_features].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# ── 5. Impute nulls with training medians ─────────────────────────────────────
# For NLP features (~24% null): listings without reviews get the training median.
# This is intentional — the model degrades gracefully for no-review listings.
fillna_medians = X_train.median(numeric_only=True).to_dict()
X_train = X_train.fillna(fillna_medians)
X_test  = X_test.fillna(fillna_medians)

# ── 6. Train ──────────────────────────────────────────────────────────────────
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100,
    random_state=RANDOM_SEED,
    class_weights={0: 1.0, 1: 1.0},   # balanced classes, no reweighting needed
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_prob)
print(f"\nSuperhost model ROC-AUC: {auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Non-superhost", "Superhost"]))

#%%
# ── 8. Export ─────────────────────────────────────────────────────────────────

# Tag each feature as actionable (host can change it) or review-based (earned)
# Used by the dashboard to generate targeted recommendations
ACTIONABLE_FEATURES = {
    "host_response_rate_clean",
    "host_acceptance_rate_clean",
    "host_has_profile_pic_num",
    "host_identity_verified_num",
    "host_response_time_num",
    "instant_bookable_num",
    "amenity_count",
    "has_wifi", "has_kitchen", "has_self_check-in", "has_parking",
    "has_air_conditioning", "has_heating", "has_tv", "has_washer",
    "has_dryer", "has_iron", "has_hair_dryer", "has_coffee",
    "has_pool", "has_gym", "has_hot_tub", "has_elevator",
}

superhost_meta = {
    "auc_score":          round(auc, 4),
    "fillna_medians":     {k: float(v) if pd.notna(v) else 0.0
                           for k, v in fillna_medians.items()},
    "actionable_features": list(ACTIONABLE_FEATURES),
    "feature_list":        superhost_features,
    # Thresholds for Low / Moderate / High potential labels
    "prob_thresholds": {
        "low":      0.40,   # < 40%  → Low potential
        "moderate": 0.65,   # 40–65% → Moderate potential
        "high":     0.65,   # > 65%  → High potential
    },
}

joblib.dump(model,            f"{MODEL_DIR}/superhost_model.pkl")
joblib.dump(superhost_features, f"{MODEL_DIR}/superhost_features.pkl")
joblib.dump(superhost_meta,   f"{MODEL_DIR}/superhost_meta.pkl")

print(f"\nSaved to {MODEL_DIR}/:")
print(f"  superhost_model.pkl      — CatBoost classifier")
print(f"  superhost_features.pkl   — {len(superhost_features)} feature names")
print(f"  superhost_meta.pkl       — AUC, fillna medians, actionable tags, thresholds")
