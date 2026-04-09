# Airbnb Investment & Host Intelligence Platform

A multi-city Airbnb market intelligence dashboard with three tabs:
- **Market Explorer** — interactive map of listings with filters and KPI cards
- **Superhost Advisor** — per-listing Superhost probability, SHAP explanations, and actionable recommendations
- **Investor Predictor** — ML-based nightly price prediction for new listings with SHAP-driven insights

Currently supports San Francisco, New York City, and Chicago. Designed to expand to additional cities without code changes.

---

## Project structure

```
airbnb_dashboard/
│
├── app.py                          ← Entry point. Run this to start the dashboard.
│
├── create_clean_dataset.py         ← Step 1: Preprocessing. Runs NLP on reviews and
│                                     engineers features from raw Inside Airbnb files.
│
├── train_investor_model.py         ← Step 2a: Trains CatBoost price regression model.
├── train_superhost_model.py        ← Step 2b: Trains CatBoost Superhost classifier.
│
├── prepare_dashboard_data.py       ← Step 3: Runs both models on all listings,
│                                     generates dashboard_listings_{city}.csv
│                                     and dashboard_meta_{city}.json.
│
├── data/
│   ├── clean_airbnb_dataset_{city}.csv    ← Output of create_clean_dataset.py
│   ├── dashboard_listings_{city}.csv      ← Output of prepare_dashboard_data.py
│   ├── dashboard_meta_{city}.json         ← Output of prepare_dashboard_data.py
│   └── review_aggregates_{city}.csv       ← NLP cache (auto-generated, speeds up reruns)
│
├── models/
│   ├── sf/
│   │   ├── price_model.pkl
│   │   ├── price_features.pkl
│   │   ├── price_meta.pkl
│   │   ├── superhost_model.pkl
│   │   ├── superhost_features.pkl
│   │   └── superhost_meta.pkl
│   ├── nyc/
│   │   └── (same structure)
│   └── chicago/
│       └── (same structure)
│
├── layouts/
│   ├── tab1_market.py              ← Market Explorer UI
│   ├── tab2_advisor.py             ← Superhost Advisor UI
│   └── tab3_predictor.py           ← Investor Predictor UI
│
└── callbacks/
    ├── market_callbacks.py         ← Tab 1 logic: filters, map, KPI cards, detail card
    ├── advisor_callbacks.py        ← Tab 2 logic: SHAP, recommendations, strengths/weaknesses
    └── predictor_callbacks.py      ← Tab 3 logic: price prediction, SHAP drivers, amenity tips
```

---

## Setup — adding a new city

Every new city follows the same four steps.

### Prerequisites

```bash
pip install dash dash-bootstrap-components plotly pandas joblib catboost scikit-learn textblob
python -m textblob.download_corpora   # first time only
```

### Step 1 — Create clean dataset

Takes raw Inside Airbnb `listings` and `reviews` CSVs and produces a single enriched CSV with engineered features and NLP review aggregates (sentiment, theme scores).

```bash
python create_clean_dataset.py \
  --city    nyc \
  --listings data/listings_nyc.csv \
  --reviews  data/reviews_nyc.csv
# Output: data/clean_airbnb_dataset_nyc.csv
# Cache:  data/review_aggregates_nyc.csv  (skip NLP on reruns)
```

The NLP step (TextBlob sentiment + theme detection) is the slow part. For a city with 400k reviews expect 10–30 minutes. The cache file means reruns are instant.

### Step 2 — Train models

Each city gets its own set of model files saved under `models/{city}/`.

```bash
# Price model (CatBoost regressor)
python train_investor_model.py \
  --city sf \
  --data data/clean_airbnb_dataset_sf.csv

# Superhost model (CatBoost classifier)
python train_superhost_model.py \
  --city sf \
  --data data/clean_airbnb_dataset_sf.csv
```

Each script saves three files: `price_model.pkl` / `superhost_model.pkl`, the feature list, and a metadata pkl containing R², fillna medians, group medians, and probability thresholds.

### Step 3 — Prepare dashboard data

Runs both models over all listings, computes market benchmarks, amenity price lifts, and neighbourhood stats. Produces the two files the dashboard reads at runtime.

```bash
python prepare_dashboard_data.py --city sf
# Output: data/dashboard_listings_sf.csv
#         data/dashboard_meta_sf.json
```

### Step 4 — Start the dashboard

```bash
python app.py
# Open: http://localhost:8050
```

The app auto-discovers cities by scanning `data/` for `dashboard_listings_{city}.csv` files — no code change needed to add a new city.

---

## How each tab works

### Tab 1 — Market Explorer

Filters (neighbourhood, room type, property type, price range, superhost status) update the Plotly OpenStreetMap scatter map in real time. Dot color indicates Superhost status (coral = Superhost, blue = non-Superhost); dot size scales with price. Clicking a dot loads a detail card on the right showing listing info, review scores, and a theme bar chart. The selected listing ID is stored in `dcc.Store` so switching to the Superhost Advisor tab auto-selects that listing.

### Tab 2 — Superhost Advisor

Select a listing from the dropdown (or arrive from a map click on Tab 1). The page shows:
- **Overview card** — price, response rate, sentiment, key amenities present/missing
- **Superhost probability** — CatBoost classifier output with Low / Moderate / High label
- **Recommendations** — up to 5 actionable items ranked by SHAP magnitude and impact tier
- **Strengths vs weaknesses** — two-column breakdown of what's working and what isn't
- **SHAP chart** — CatBoost native TreeSHAP showing which features pushed this listing toward or away from Superhost status, with green/red bars and +/− direction labels

SHAP values are computed per-listing at click time using CatBoost's built-in `type="ShapValues"` — no external SHAP library needed.

### Tab 3 — Investor Predictor

Input form for a planned listing (neighbourhood, property type, bedrooms, bathrooms, amenities, host setup). Clicking Predict runs the CatBoost price model and returns:
- **Predicted nightly price** with ±12% estimated range
- **Market comparison** — predicted price vs neighbourhood median and property type median
- **Key price drivers** — SHAP-based waterfall bars showing each feature's dollar contribution
- **Investor tips** — amenity gap analysis using observed market lift data from `dashboard_meta_{city}.json`

The form derives coordinates from neighbourhood centroids, infers room type flags from property type, and fills `availability_60` and `maximum_nights` with training-set medians — inputs a future investor can't know.

---

## Multi-city design

All three callbacks load every city's data and models at startup into in-memory dicts (`ALL_DF`, `ALL_PRICE_MODELS`, `ALL_SH_MODELS`). A `dcc.Store(id="selected-city")` holds the active city and is updated by the city dropdown in the navbar. Each callback reads `State("selected-city", "data")` to pick the right dataframe and model without reloading from disk.

Adding a new city requires no code changes — just run the four steps above and restart the app.

---

## Key technical decisions

**CatBoost native SHAP** — both models use `model.get_feature_importance(data=Pool(X), type="ShapValues")` for local explanations. This is CatBoost's built-in TreeSHAP implementation, so no external `shap` library is required.

**No price filter on dashboard listings** — `prepare_dashboard_data.py` includes all listings regardless of whether `price_clean` is null. The price cap (99th percentile) is only used for computing benchmark stats, not for filtering rows. This ensures every listing in the dataset appears in the Advisor dropdown.

**Group median encoding at prediction time** — the price model was trained with leak-free neighbourhood and property type median encoding. The same `group_medians` dict from `price_meta.pkl` is applied at prediction time to avoid train/serve skew.

**Review NLP caching** — `create_clean_dataset.py` saves `review_aggregates_{city}.csv` after the first run. Subsequent runs load from cache, skipping the slow TextBlob step entirely.
