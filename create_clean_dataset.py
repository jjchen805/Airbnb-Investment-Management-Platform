"""
create_clean_dataset.py
=======================
Produces a clean, model-ready dataset from raw Inside Airbnb files.
Works for any city — outputs clean_airbnb_dataset_{city}.csv.

Usage:
    python create_clean_dataset.py --city sf --listings data/listings_sf.csv --reviews data/reviews_sf.csv
    python create_clean_dataset.py --city nyc --listings data/listings_nyc.csv --reviews data/reviews_nyc.csv

Output:
    data/clean_airbnb_dataset_{city}.csv

The NLP step (sentiment + theme detection) is slow for large review files.
A cache file is saved to data/review_aggregates_{city}.csv after the first run
so re-runs skip the NLP entirely.

Dependencies:
    pip install pandas numpy textblob
    python -m textblob.download_corpora   # first time only
"""

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Constants ─────────────────────────────────────────────────────────────────

# Themes to detect in review text — same set used across all cities
THEME_KEYWORDS = {
    "cleanliness":   ["clean", "dirty", "spotless", "tidy", "dust", "stain", "hygien", "sanit"],
    "communication": ["communicat", "responsive", "reply", "response", "message", "contact", "reach"],
    "checkin":       ["check-in", "checkin", "check in", "key", "lockbox", "arrival", "access"],
    "location":      ["location", "neighborhood", "neighbourhood", "area", "walk", "transport", "bus", "train", "bart", "subway", "metro"],
    "amenities":     ["ameniti", "kitchen", "wifi", "towel", "soap", "coffee", "washer", "dryer", "parking"],
    "accuracy":      ["accurate", "description", "photo", "picture", "expect", "as described", "mislead"],
    "noise":         ["noise", "noisy", "quiet", "loud", "sound", "peaceful"],
    "comfort":       ["comfort", "cozy", "bed", "mattress", "pillow", "sleep", "rest"],
    "value":         ["value", "price", "worth", "expensive", "cheap", "affordable", "deal", "overpriced"],
}

# Amenities to flag — keyword matched against the raw amenities string
KEY_AMENITIES = [
    "wifi", "kitchen", "washer", "dryer", "parking",
    "air conditioning", "heating", "tv", "iron", "hair dryer",
    "smoke alarm", "carbon monoxide alarm", "first aid kit",
    "fire extinguisher", "self check-in", "coffee",
    "pool", "hot tub", "gym", "elevator",
]

# Top N neighbourhoods to keep as named categories; rest → "Other"
TOP_NEIGHBOURHOODS = 20

# Top N property types to keep as named categories; rest → "Other"
TOP_PROPERTY_TYPES = 10


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def clean_price(series: pd.Series) -> pd.Series:
    return (series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .apply(pd.to_numeric, errors="coerce"))


def clean_percentage(series: pd.Series) -> pd.Series:
    return (series.astype(str)
            .str.replace("%", "", regex=False)
            .apply(pd.to_numeric, errors="coerce"))


def parse_boolean(series: pd.Series) -> pd.Series:
    return series.map({"t": 1, "f": 0, 1.0: 1, 0.0: 0, True: 1, False: 0})


def parse_bathrooms_text(text) -> float:
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    if "half" in text:
        return 0.5
    match = re.search(r"(\d+\.?\d*)", text)
    return float(match.group(1)) if match else np.nan


def count_amenities(amenities_str) -> int:
    if pd.isna(amenities_str) or amenities_str == "[]":
        return 0
    try:
        return len(re.findall(r'"([^"]*)"', str(amenities_str)))
    except Exception:
        return 0


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_data(listings_path: str, reviews_path: str):
    print("Loading raw data...")
    listings = pd.read_csv(listings_path)
    reviews  = pd.read_csv(reviews_path)
    print(f"  Listings : {len(listings):,} rows, {len(listings.columns)} columns")
    print(f"  Reviews  : {len(reviews):,} rows")
    return listings, reviews


# ── Step 2: NLP review aggregation ───────────────────────────────────────────

def aggregate_reviews(reviews: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    """
    Run sentiment analysis and theme detection on review comments.
    Results are cached to cache_path so re-runs are instant.
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached review aggregates from {cache_path}")
        return pd.read_csv(cache_path)

    print("  Running NLP on reviews (this takes a few minutes for large datasets)...")
    from textblob import TextBlob

    reviews_clean = reviews.dropna(subset=["comments"]).copy()
    reviews_clean["comments"] = reviews_clean["comments"].astype(str)

    # Sentiment
    def get_sentiment(text):
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception:
            return 0.0, 0.5

    print(f"    Analysing sentiment for {len(reviews_clean):,} reviews...")
    sentiments = reviews_clean["comments"].apply(get_sentiment)
    reviews_clean["sentiment_polarity"]     = sentiments.apply(lambda x: x[0])
    reviews_clean["sentiment_subjectivity"] = sentiments.apply(lambda x: x[1])

    # Theme flags
    for theme, keywords in THEME_KEYWORDS.items():
        pattern = "|".join(keywords)
        reviews_clean[f"theme_{theme}"] = (
            reviews_clean["comments"].str.lower()
            .str.contains(pattern, regex=True)
            .astype(int)
        )
        # Positive and negative variants
        reviews_clean[f"theme_{theme}_positive"] = (
            (reviews_clean[f"theme_{theme}"] == 1) &
            (reviews_clean["sentiment_polarity"] > 0.1)
        ).astype(int)
        reviews_clean[f"theme_{theme}_negative"] = (
            (reviews_clean[f"theme_{theme}"] == 1) &
            (reviews_clean["sentiment_polarity"] < -0.1)
        ).astype(int)

    # Review length
    reviews_clean["review_length"] = reviews_clean["comments"].str.len()

    # Aggregate to listing level
    theme_cols = [c for c in reviews_clean.columns if c.startswith("theme_")]
    agg_dict = {
        "id":                       "count",
        "sentiment_polarity":       ["mean", "std", "min"],
        "sentiment_subjectivity":   "mean",
        "review_length":            ["mean", "std"],
    }
    for col in theme_cols:
        agg_dict[col] = "mean"

    agg = reviews_clean.groupby("listing_id").agg(agg_dict)
    agg.columns = ["_".join(col).strip("_") for col in agg.columns]
    agg = agg.rename(columns={"id_count": "review_count_from_text"})
    agg = agg.reset_index()

    agg.to_csv(cache_path, index=False)
    print(f"  Cached review aggregates → {cache_path}")
    print(f"  Listings with review data: {len(agg):,}")
    return agg


# ── Step 3: Feature engineering on listings ───────────────────────────────────

def engineer_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered columns to the listings dataframe."""
    df = listings.copy()

    # Reference date derived from the scrape — works for any city
    reference_date = pd.to_datetime(df["last_scraped"]).max()
    print(f"  Reference date (from last_scraped): {reference_date.date()}")

    # ── Price ──────────────────────────────────────────────────────────────
    df["price_clean"] = clean_price(df["price"])

    # ── Host fields ────────────────────────────────────────────────────────
    df["host_is_superhost"]       = parse_boolean(df["host_is_superhost"])
    df["host_response_rate_clean"]  = clean_percentage(df.get("host_response_rate", pd.Series(dtype=str)))
    df["host_acceptance_rate_clean"]= clean_percentage(df.get("host_acceptance_rate", pd.Series(dtype=str)))
    df["host_has_profile_pic_num"]  = parse_boolean(df.get("host_has_profile_pic", pd.Series(dtype=str)))
    df["host_identity_verified_num"]= parse_boolean(df.get("host_identity_verified", pd.Series(dtype=str)))
    df["instant_bookable_num"]      = parse_boolean(df.get("instant_bookable", pd.Series(dtype=str)))
    df["has_availability_num"]      = parse_boolean(df.get("has_availability", pd.Series(dtype=str)))

    # Host response time: 1 = within an hour (best), 4 = few days (worst)
    response_time_map = {
        "within an hour":      1,
        "within a few hours":  2,
        "within a day":        3,
        "a few days or more":  4,
    }
    df["host_response_time_num"] = df.get("host_response_time", pd.Series(dtype=str)).map(response_time_map)

    # ── Date features ──────────────────────────────────────────────────────
    for date_col in ["host_since", "first_review", "last_review"]:
        if date_col in df.columns:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            df[f"{date_col}_parsed"] = parsed
            df[f"{date_col}_days"]   = (reference_date - parsed).dt.days

    df["host_tenure_years"]      = df.get("host_since_days", pd.Series(dtype=float)) / 365.25
    df["days_since_last_review"] = df.get("last_review_days", pd.Series(dtype=float))

    # ── Bathrooms ──────────────────────────────────────────────────────────
    df["bathrooms_clean"] = df.get("bathrooms_text", pd.Series(dtype=str)).apply(parse_bathrooms_text)
    if "bathrooms" in df.columns:
        df["bathrooms_clean"] = df["bathrooms_clean"].fillna(pd.to_numeric(df["bathrooms"], errors="coerce"))

    # ── Amenities ──────────────────────────────────────────────────────────
    df["amenity_count"] = df["amenities"].apply(count_amenities)

    for amenity in KEY_AMENITIES:
        col = f"has_{amenity.replace(' ', '_')}"
        df[col] = df["amenities"].apply(
            lambda x: int(amenity.lower() in str(x).lower()) if pd.notna(x) else 0
        )

    # ── Room type flags ────────────────────────────────────────────────────
    df["is_entire_home"]  = (df["room_type"] == "Entire home/apt").astype(int)
    df["is_private_room"] = (df["room_type"] == "Private room").astype(int)
    df["is_shared_room"]  = (df["room_type"] == "Shared room").astype(int)

    # ── Neighbourhood bucketing ────────────────────────────────────────────
    # Keep top N as named, rest → "Other" so models across cities are consistent
    top_nbhds = df["neighbourhood_cleansed"].value_counts().head(TOP_NEIGHBOURHOODS).index
    df["neighbourhood_top"] = df["neighbourhood_cleansed"].apply(
        lambda x: x if x in top_nbhds else "Other"
    )

    # ── Property type simplification ──────────────────────────────────────
    top_props = df["property_type"].value_counts().head(TOP_PROPERTY_TYPES).index
    df["property_type_simple"] = df["property_type"].apply(
        lambda x: x if x in top_props else "Other"
    )

    return df


# ── Step 4: Merge listings + review aggregates ────────────────────────────────

def merge_and_finalise(df: pd.DataFrame, review_agg: pd.DataFrame) -> pd.DataFrame:
    """Merge engineered listings with NLP review aggregates."""
    df = df.merge(
        review_agg,
        left_on="id", right_on="listing_id",
        how="left",
    )
    df["_index"] = df.index
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create clean Airbnb dataset for a city.")
    parser.add_argument("--city",     required=True,  help="City code e.g. sf, nyc, la")
    parser.add_argument("--listings", required=True,  help="Path to listings CSV")
    parser.add_argument("--reviews",  required=True,  help="Path to reviews CSV")
    parser.add_argument("--out-dir",  default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    city     = args.city.lower().replace(" ", "_")
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    out_path   = os.path.join(out_dir, f"clean_airbnb_dataset_{city}.csv")
    cache_path = os.path.join(out_dir, f"review_aggregates_{city}.csv")

    print("=" * 60)
    print(f"Creating clean dataset for city: {city.upper()}")
    print("=" * 60)

    # Step 1: Load
    listings, reviews = load_data(args.listings, args.reviews)

    # Step 2: NLP review aggregation (cached after first run)
    print("\nStep 2: Review NLP aggregation")
    review_agg = aggregate_reviews(reviews, cache_path)

    # Step 3: Feature engineering
    print("\nStep 3: Feature engineering")
    df = engineer_features(listings)

    # Step 4: Merge
    print("\nStep 4: Merging listings + review aggregates")
    df_final = merge_and_finalise(df, review_agg)

    # Save
    df_final.to_csv(out_path, index=False)

    print()
    print("=" * 60)
    print(f"✓  Done.")
    print(f"   Output : {out_path}")
    print(f"   Rows   : {len(df_final):,}")
    print(f"   Columns: {len(df_final.columns)}")
    print(f"   Listings with review data : {df_final['listing_id'].notna().sum():,}")
    print(f"   Listings without reviews  : {df_final['listing_id'].isna().sum():,}")
    print(f"   Superhost rate            : {df_final['host_is_superhost'].mean()*100:.1f}%")
    print(f"   Price null rate           : {df_final['price_clean'].isna().mean()*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
