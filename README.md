# Airbnb Investment & Host Intelligence Platform
 
A multi-city Airbnb market intelligence dashboard built with Dash, CatBoost, and SHAP. Designed as a reusable framework — any city available on [Inside Airbnb](https://insideairbnb.com) can be onboarded in four commands.
 
---
 
## What it does
 
The app opens on a landing page where users identify themselves as one of three types, then get directed to the relevant tab:
 
- **🗺️ Market Explorer** — interactive Mapbox map of all listings with filters (neighbourhood, room type, property type, price range, Superhost status), KPI summary cards, and a listing detail panel with review theme bars color-coded by sentiment direction
- **🌟 Superhost Advisor** — select any existing listing to see its Superhost probability (CatBoost classifier), a SHAP-driven explanation, ranked actionable recommendations, strengths vs weaknesses breakdown, and an AI action plan powered by Claude
- **📈 Investor Predictor** — input form for a planned listing that returns an ML-based nightly price estimate, market comparison, SHAP price drivers, amenity gap tips, what-if upgrade scenarios, and an AI investment brief
City switching is available in the navbar — all three tabs update instantly. Adding a new city requires no code changes.
 
---
 
## Project structure
 
```
airbnb_dashboard/
│
├── app.py                          ← Entry point. Run this to start the dashboard.
│
├── create_clean_dataset.py         ← Step 1: Preprocessing. Runs NLP on reviews
│                                     and engineers features from raw Inside Airbnb files.
│
├── train_investor_model.py         ← Step 2a: Trains CatBoost price regression model.
├── train_superhost_model.py        ← Step 2b: Trains CatBoost Superhost classifier.
│
├── prepare_dashboard_data.py       ← Step 3: Runs both models on all listings,
│                                     generates dashboard_listings_{city}.csv
│                                     and dashboard_meta_{city}.json.
│
├── data/
│   ├── listings_{city}.csv                ← Raw Inside Airbnb listings
│   ├── reviews_{city}.csv                 ← Raw Inside Airbnb reviews
│   ├── clean_airbnb_dataset_{city}.csv    ← Output of create_clean_dataset.py
│   ├── review_aggregates_{city}.csv       ← NLP cache (auto-generated, speeds up reruns)
│   ├── dashboard_listings_{city}.csv      ← Output of prepare_dashboard_data.py
│   └── dashboard_meta_{city}.json         ← Output of prepare_dashboard_data.py
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
│   ├── home.py                     ← Landing page with user type selector
│   ├── tab1_market.py              ← Market Explorer UI
│   ├── tab2_advisor.py             ← Superhost Advisor UI
│   ├── tab3_predictor.py           ← Investor Predictor UI
│   └── chat.py                     ← Floating chat assistant UI
│
├── callbacks/
│   ├── market_callbacks.py         ← Tab 1: filters, map, KPI cards, detail card
│   ├── advisor_callbacks.py        ← Tab 2: SHAP, recommendations, AI advisor
│   ├── predictor_callbacks.py      ← Tab 3: price prediction, what-if engine, AI brief
│   └── chat_callbacks.py           ← Floating chat: market Q&A via Anthropic API
│
├── services/
│   └── llm_agent.py                ← Multi-provider AI agent (Anthropic/OpenAI/NVIDIA)
│                                     with LRU caching, threading, and timeout logic
│
└── assets/
    └── style.css                   ← Global design system (Apple-inspired typography,
                                      colors, spacing, card styles)
```
 
---
 
## Adding a new city
 
Every city follows the same four steps. Inside Airbnb provides `listings.csv` and `reviews.csv` for 100+ cities worldwide — all in the same schema.
 
### Prerequisites
 
```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # first time only
```
 
### Step 1 — Create clean dataset
 
```bash
python create_clean_dataset.py \
  --city    tokyo \
  --listings data/listings_tokyo.csv \
  --reviews  data/reviews_tokyo.csv
# Output : data/clean_airbnb_dataset_tokyo.csv
# Cache  : data/review_aggregates_tokyo.csv  (skips NLP on reruns)
```
 
The NLP step (TextBlob sentiment + 9 review theme detectors) is the slow part — allow 10–30 minutes for large cities. The cache file means subsequent reruns are instant.
 
### Step 2 — Train models
 
```bash
python train_investor_model.py \
  --city tokyo \
  --data data/clean_airbnb_dataset_tokyo.csv
 
python train_superhost_model.py \
  --city tokyo \
  --data data/clean_airbnb_dataset_tokyo.csv
# Output: models/tokyo/ with 6 pkl files
```
 
Expected model performance based on SF / NYC / Chicago:
- Price model R² ~0.65–0.75
- Superhost classifier ROC-AUC ~0.86–0.89
### Step 3 — Prepare dashboard data
 
```bash
python prepare_dashboard_data.py --city tokyo
# Output: data/dashboard_listings_tokyo.csv
#         data/dashboard_meta_tokyo.json
```
 
### Step 4 — Start the app
 
```bash
python app.py
# Open: http://localhost:8050
# Tokyo now appears automatically in the city dropdown
```
 
The app auto-discovers cities by scanning `data/` for `dashboard_listings_{city}.csv` files — no code changes needed.
 
---
 
## Running the app
 
```bash
# Activate your virtual environment
source .venv/bin/activate
 
# From the project root
python app.py
# Open http://localhost:8050
```
 
If you see a `NotImplementedError: Cannot` on import, you are using the Anaconda environment instead of the venv. Always activate the venv first.
 
---
 
## Environment variables
 
Create a `.env` file in the project root (already in `.gitignore`):
 
```
# Mapbox — required for the map
MAPBOX_TOKEN=pk.eyJ1...
 
# AI agent — required for Tab 2 and Tab 3 AI features and floating chat
ENABLE_AGENT=true
AGENT_PROVIDER=anthropic          # anthropic | openai | nvidia
ANTHROPIC_API_KEY=sk-ant-...      # if AGENT_PROVIDER=anthropic
OPENAI_API_KEY=sk-...             # if AGENT_PROVIDER=openai
NVIDIA_API_KEY=...                # if AGENT_PROVIDER=nvidia (free tier, slower)
 
# Optional overrides
AGENT_MODEL=claude-haiku-4-5      # override default model for chosen provider
AGENT_TIMEOUT_SECONDS=15
```
 
To switch providers, change `AGENT_PROVIDER` and the corresponding key — no code changes needed.
 
---
 
## How each tab works
 
### Landing page
 
Opens on load with three user-type cards. Clicking a card navigates to the relevant tab and reveals the navbar (with city selector and a Home back button). Implemented using persistent hidden buttons and a clientside callback to work around Dash's requirement that all callback Inputs exist in the DOM at all times.
 
### Tab 1 — Market Explorer
 
Filters update a Plotly `scatter_mapbox` map in real time using the Mapbox `streets-v12` style. Dots are colored by Superhost status (blue = Superhost, warm beige = non-Superhost) at 0.55 opacity. Dot size scales with price, capped at $800. Clicking a dot loads a detail card with listing info and review theme bars. Theme bar colors reflect sentiment direction — green means guests praised that aspect, orange means mixed, red means complaints — using `theme_{theme}_positive_mean` minus `theme_{theme}_negative_mean`. The selected listing ID is stored in `dcc.Store` so switching to the Superhost Advisor auto-selects that listing.
 
### Tab 2 — Superhost Advisor
 
Select a listing from the dropdown or arrive via a map click on Tab 1. The page runs CatBoost's native TreeSHAP on the selected listing and returns:
- Superhost probability with Low / Moderate / High potential label
- Up to 5 recommendations ranked by impact tier then SHAP magnitude
- Strengths vs weaknesses breakdown
- SHAP waterfall chart (green = pushes toward Superhost, red = pushes away)
- **AI Advisor panel** — generate a 7/30-day action plan or ask any question about the listing, grounded in its SHAP values and recommendations
Recommendations are only surfaced when a feature is both weak or missing AND has a negative SHAP value for that listing — not generic advice.
 
### Tab 3 — Investor Predictor
 
Form inputs are mapped to the exact feature pipeline used during training: group median encoding from `price_meta.pkl`, neighbourhood centroid coordinates, and room type flags derived from property type. Fields a future investor cannot know (`availability_60`, `maximum_nights`) are filled with training-set medians. Output panel shows:
- Predicted nightly price with estimated range
- Market comparison vs neighbourhood and property type medians
- SHAP price drivers with dollar-impact values
- Investor tips grounded in observed market lift data per neighbourhood
- **What-if scenarios** — runs 15–20 real model predictions across candidate upgrades within a user-specified budget, ranked by price uplift and booking competitiveness
- **AI Investment Advisor** — generate an investment brief, interpret what-if results, or ask any question about the prediction
### Floating chat assistant
 
A chat button in the bottom right corner opens a slide-in drawer on every page. Answers general market questions (neighbourhood comparisons, amenity lifts, Superhost rates) grounded in `dashboard_meta_{city}.json` for the selected city. Conversation history is maintained within the session via `dcc.Store`.
 
---
 
## AI agent (`services/llm_agent.py`)
 
Supports three providers switchable via `.env` with no code changes:
 
| Provider | Default model | Speed | Cost |
|---|---|---|---|
| Anthropic (recommended) | `claude-haiku-4-5` | ~1–2s | ~$0.001/call |
| OpenAI | `gpt-4.1-mini` | ~2–3s | ~$0.002/call |
| NVIDIA | `moonshotai/kimi-k2.5` | ~8–15s | Free tier |
 
All agent functions share:
- **LRU cache** — identical inputs return instantly without an API call
- **Thread + hard timeout** — API calls never block the Dash UI
- **Graceful degradation** — returns `None` on any failure; UI shows a fallback message
---
 
## Multi-city design
 
All three callbacks preload every city's data and models into memory at startup (`ALL_DF`, `ALL_PRICE_MODELS`, `ALL_SH_MODELS`). A `dcc.Store(id="selected-city")` holds the active city and is updated by the navbar dropdown. Each callback reads `State("selected-city", "data")` to pick the right dataframe and model without reloading from disk. Adding a new city only requires running the four pipeline steps — the app discovers it automatically on next restart.
 
---
 
## Key technical decisions
 
**CatBoost native SHAP** — both models use `model.get_feature_importance(data=Pool(X), type="ShapValues")`. No external `shap` library required.
 
**No price filter on dashboard listings** — `prepare_dashboard_data.py` includes all listings regardless of whether `price_clean` is null. The 99th percentile cap is only used for computing benchmark stats, not filtering rows. This ensures every listing appears in the Advisor dropdown.
 
**Review theme sentiment coloring** — theme bar colors use net sentiment direction (positive minus negative mention rate) rather than raw mention frequency. This makes red mean guests complained rather than guests rarely mentioned it.
 
**Group median encoding at prediction time** — the price model was trained with leak-free neighbourhood and property type median encoding. The same `group_medians` dict from `price_meta.pkl` is applied at prediction time to avoid train/serve skew.
 
**What-if scenario engine** — purely ML-driven with no LLM involved. Generates candidate upgrades (single amenities, bundles, host setup changes, minimum nights variants), filters by budget, reruns the price model for each candidate, and ranks by a composite objective (price uplift + booking competitiveness proxy). The LLM then interprets the top 3 results as a narrative.
 
**NLP caching** — `create_clean_dataset.py` saves `review_aggregates_{city}.csv` after the first run. Subsequent runs skip the slow TextBlob step entirely.
 
**Persistent hidden buttons** — the home page navigation uses persistent hidden `dbc.Button` elements and a clientside callback that mirrors visible card button clicks to them. This works around Dash's requirement that all callback Inputs exist in the DOM at all times.
 
---
 
## Git
 
Data and model files are excluded from version control:
 
```gitignore
data/
models/
.env
__pycache__/
*.pyc
.DS_Store
```
 
All excluded files are regenerated locally by running the four pipeline steps above.