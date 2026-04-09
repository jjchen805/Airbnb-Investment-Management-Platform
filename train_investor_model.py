import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--city", required=True, help="e.g. sf, nyc, chicago")
parser.add_argument("--data", required=True, help="Path to clean_airbnb_dataset_{city}.csv")
args = parser.parse_args()

# ── 0. Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = f"models/{args.city}"
DATA_PATH = args.data
os.makedirs(MODEL_DIR, exist_ok=True) 
PRICE_CAP   = 0.99          # quantile cap for outlier removal
TEST_SIZE   = 0.2
RANDOM_SEED = 42
 
# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
 
# ── 2. Target — filter nulls and outliers ─────────────────────────────────────
y = pd.to_numeric(df['price_clean'], errors='coerce')
valid_idx = y.notna()
df = df[valid_idx].copy()
y = y[valid_idx]
 
price_cap_value = float(y.quantile(PRICE_CAP))
mask = y <= price_cap_value
df, y = df[mask].copy(), y[mask]
 
print(f"Training rows after outlier cap (${price_cap_value:.0f}): {len(df)}")
 
# ── 3. Feature list ────────────────────────────────────────────────────────────
investor_features = [
    # Property basics
    'accommodates', 'bathrooms_clean', 'bedrooms', 'beds',

    # Booking constraints
    'minimum_nights', 'maximum_nights',

    # Availability
    'availability_60', 'availability_365',

    # Host setup (investor can configure)
    'instant_bookable_num', 'host_is_superhost', 'calculated_host_listings_count',

    # Core amenities
    'amenity_count', 'has_wifi', 'has_kitchen', 'has_washer', 'has_dryer',
    'has_parking', 'has_air_conditioning', 'has_heating', 'has_tv', 'has_self_check-in',

    # Premium amenities (high signal!)
    'has_gym', 'has_pool', 'has_elevator', 'has_iron',
    'has_coffee', 'has_hair_dryer', 'has_hot_tub',

    # Room type
    'is_entire_home', 'is_private_room', 'is_shared_room',

    # Location (add coordinates — very important)
    'latitude', 'longitude',
    'neighbourhood_top', 'property_type_simple',
]

investor_features = [f for f in investor_features if f in df.columns]

# ── 4. Train / test split (BEFORE any target encoding) ────────────────────────
extra_cols = [c for c in ['neighbourhood_top', 'property_type_simple', 'price_clean']
              if c not in investor_features]  # only add if not already present
X_raw = df[investor_features + extra_cols].copy()
y_log = np.log1p(y)

X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X_raw, y_log, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
 
# ── 5. Leak-free group median encoding ────────────────────────────────────────
group_medians = {}
for col in ['neighbourhood_top', 'property_type_simple']:
    medians = X_tr_raw.groupby(col)['price_clean'].median()
    group_medians[col] = medians.to_dict()
    X_tr_raw[f'{col}_median_price'] = X_tr_raw[col].map(medians)
    X_te_raw[f'{col}_median_price'] = X_te_raw[col].map(medians)
 
# Drop price_clean helper column (was only needed for encoding)
X_tr_raw = X_tr_raw.drop(columns=['price_clean'])
X_te_raw = X_te_raw.drop(columns=['price_clean'])
 
# ── 6. One-hot encode categoricals ────────────────────────────────────────────
cat_cols = [c for c in ['neighbourhood_top', 'property_type_simple'] if c in X_tr_raw.columns]
X_train = pd.get_dummies(X_tr_raw, columns=cat_cols, drop_first=True)
X_test  = pd.get_dummies(X_te_raw, columns=cat_cols, drop_first=True)
 
# Align test to train columns (fills unseen dummies with 0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
 
# ── 7. Impute remaining nulls with training medians ───────────────────────────
fillna_medians = X_train.median(numeric_only=True).to_dict()
X_train = X_train.fillna(fillna_medians)
X_test  = X_test.fillna(fillna_medians)
 
# ── 8. Train ──────────────────────────────────────────────────────────────────
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=RANDOM_SEED,
)
model.fit(X_train, y_train)
 
# ── 9. Evaluate ──────────────────────────────────────────────────────────────
y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)
r2 = r2_score(y_true, y_pred)
print(f"Investor model R²: {r2:.4f}")

# ── 10. Export ────────────────────────────────────────────────────────────────
price_meta = {
    "price_cap_value":  price_cap_value,
    "log_target":       True,           # predict() output needs np.expm1()
    "group_medians":    group_medians,  # neighbourhood_top & property_type_simple
    "fillna_medians":   fillna_medians, # imputation values from training set
    "r2_score":         round(r2, 4),  # for reference in dashboard
}
 
joblib.dump(model,                  f"{MODEL_DIR}/price_model.pkl")
joblib.dump(X_train.columns.tolist(), f"{MODEL_DIR}/price_features.pkl")
joblib.dump(price_meta,             f"{MODEL_DIR}/price_meta.pkl")
 
print(f"\nSaved to {MODEL_DIR}/:")
print(f"  price_model.pkl      — CatBoost model")
print(f"  price_features.pkl   — {len(X_train.columns)} columns (post-dummies order)")
print(f"  price_meta.pkl       — cap, group medians, fillna values")
