# %%
# ============================================
# preprocess_airbnb.py
# - Cleans listings + reviews
# - Saves processed CSVs
# - Merges on reviews.listing_id == listings.id
#
# INPUT (uploaded):
#   /data/raw/listings_sf.csv
#   /data/raw/reviews_sf.csv
#
# OUTPUT:
#   ./data/processed/listings_clean.csv
#   ./data/processed/reviews_clean.csv
#   ./data/processed/reviews_with_listings.csv
# ============================================
#%%
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

#%%
# -------------- Config --------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LISTINGS_PATH = "/Users/joshuachen/Projects/Airbnb-Investment-Management-Platform/data/raw/listings_sf.csv"
REVIEWS_PATH = "/Users/joshuachen/Projects/Airbnb-Investment-Management-Platform/data/raw/reviews_sf.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#%%
# -------------- Helpers --------------
def _coerce_id_series(s: pd.Series) -> pd.Series:
    """
    Coerce IDs like '12345' or 12345.0 -> Int64.
    Keeps missing as <NA>.
    """
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _clean_price_to_float(s: pd.Series) -> pd.Series:
    """
    Handles strings like '$1,234.00' or '1234' -> float.
    Leaves missing as NaN.
    """
    if s.dtype != "object":
        # already numeric-ish
        return pd.to_numeric(s, errors="coerce").astype(float)

    cleaned = (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _safe_lower(s: pd.Series) -> pd.Series:
    return s.astype("string").str.lower()


def _standardize_missing_text(s: pd.Series) -> pd.Series:
    """
    Convert empty strings and known placeholders to <NA>.
    """
    s = s.astype("string")
    s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "None": pd.NA})
    return s


def _impute_numeric_with_indicator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Add <col>_missing indicator, then median-impute numeric.
    """
    if col not in df.columns:
        return df
    miss_col = f"{col}_missing"
    df[miss_col] = df[col].isna().astype("int8")
    med = df[col].median(skipna=True)
    df[col] = df[col].fillna(med)
    return df


# -------------- Listings cleaning --------------
def preprocess_listings(listings: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    listings.columns = [c.strip() for c in listings.columns]

    # Required keys
    if "id" not in listings.columns:
        raise ValueError("Expected listings to have an 'id' column.")

    listings["id"] = _coerce_id_series(listings["id"])
    listings = listings.dropna(subset=["id"]).copy()

    # Common columns (only apply if present)
    if "host_id" in listings.columns:
        listings["host_id"] = _coerce_id_series(listings["host_id"])

    # Price (many InsideAirbnb listings have "price")
    if "price" in listings.columns:
        listings["price"] = _clean_price_to_float(listings["price"])
        listings = _impute_numeric_with_indicator(listings, "price")

    # Lat/Lon
    for col in ["latitude", "longitude"]:
        if col in listings.columns:
            listings[col] = pd.to_numeric(listings[col], errors="coerce")

    # Bedrooms/bathrooms/accommodates etc.
    numeric_candidates = [
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms",
        "bathrooms_text",  # may be text, ignore later
        "minimum_nights",
        "maximum_nights",
        "availability_30",
        "availability_60",
        "availability_90",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ]
    for col in numeric_candidates:
        if col in listings.columns:
            # bathrooms_text might be non-numeric; coercing is okay
            listings[col] = pd.to_numeric(listings[col], errors="coerce")

    # Impute some core numeric fields
    for col in ["bedrooms", "beds", "accommodates", "minimum_nights", "review_scores_rating"]:
        if col in listings.columns:
            listings = _impute_numeric_with_indicator(listings, col)

    # Categorical fills
    cat_fill = {
        "neighbourhood": "Unknown",
        "neighbourhood_cleansed": "Unknown",
        "room_type": "Unknown",
        "property_type": "Unknown",
    }
    for col, fillv in cat_fill.items():
        if col in listings.columns:
            listings[col] = _standardize_missing_text(listings[col]).fillna(fillv)

    # Amenities as string (keep as-is for now, parse later)
    if "amenities" in listings.columns:
        listings["amenities"] = _standardize_missing_text(listings["amenities"])

    # Ensure there are no duplicate listing IDs
    listings = listings.drop_duplicates(subset=["id"]).copy()

    return listings


# -------------- Reviews cleaning --------------
def preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews.columns = [c.strip() for c in reviews.columns]

    # Required keys
    # Some versions use "listing_id", some use "listing_id" exactly
    if "listing_id" not in reviews.columns:
        raise ValueError("Expected reviews to have a 'listing_id' column.")

    reviews["listing_id"] = _coerce_id_series(reviews["listing_id"])
    reviews = reviews.dropna(subset=["listing_id"]).copy()

    # Comments text
    # InsideAirbnb typically uses "comments"
    if "comments" not in reviews.columns:
        raise ValueError("Expected reviews to have a 'comments' column.")
    reviews["comments"] = _standardize_missing_text(reviews["comments"])
    reviews = reviews.dropna(subset=["comments"]).copy()
    reviews["comments"] = _safe_lower(reviews["comments"])

    # Date (optional)
    if "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    # Optional light features (helpful for QA + baseline models)
    reviews["review_length"] = reviews["comments"].str.len().astype("Int64")

    return reviews


# -------------- Merge --------------
def merge_reviews_with_listings(
    reviews: pd.DataFrame, listings: pd.DataFrame, how: str = "inner"
) -> pd.DataFrame:
    """
    Merge reviews with listings on reviews.listing_id == listings.id.
    """
    merged = reviews.merge(
        listings,
        left_on="listing_id",
        right_on="id",
        how=how,
        suffixes=("_review", "_listing"),
        validate="many_to_one",  # many reviews per listing
    )
    return merged


# -------------- Main --------------
def main() -> None:
    # Load
    listings_raw = pd.read_csv(LISTINGS_PATH, low_memory=False)
    reviews_raw = pd.read_csv(REVIEWS_PATH, low_memory=False)

    # Clean
    listings_clean = preprocess_listings(listings_raw)
    reviews_clean = preprocess_reviews(reviews_raw)

    # Filter reviews to valid listing ids (saves memory)
    valid_ids = set(listings_clean["id"].dropna().astype(int).tolist())
    reviews_clean = reviews_clean[reviews_clean["listing_id"].dropna().astype(int).isin(valid_ids)].copy()

    # Merge (optional; for NLP you might not need this full merge)
    merged = merge_reviews_with_listings(reviews_clean, listings_clean, how="inner")

    # Save
    listings_out = OUT_DIR / "listings_clean.csv"
    reviews_out = OUT_DIR / "reviews_clean.csv"
    merged_out = OUT_DIR / "reviews_with_listings.csv"

    listings_clean.to_csv(listings_out, index=False)
    reviews_clean.to_csv(reviews_out, index=False)
    merged.to_csv(merged_out, index=False)

    # Quick QA prints
    print("Saved:")
    print(f"- {listings_out} | rows={len(listings_clean):,} cols={listings_clean.shape[1]}")
    print(f"- {reviews_out}   | rows={len(reviews_clean):,} cols={reviews_clean.shape[1]}")
    print(f"- {merged_out}    | rows={len(merged):,} cols={merged.shape[1]}")
    print()
    print("Missingness snapshot (listings):")
    miss = listings_clean.isna().mean().sort_values(ascending=False).head(15)
    print(miss.to_string())
    print()
    print("Missingness snapshot (reviews):")
    miss_r = reviews_clean.isna().mean().sort_values(ascending=False).head(10)
    print(miss_r.to_string())


if __name__ == "__main__":
    main()

'''
Missingness snapshot (listings):
bathrooms_text                  1.000000
calendar_updated                1.000000
neighbourhood_group_cleansed    1.000000
neighborhood_overview           0.392619
host_about                      0.385828
license                         0.343157
estimated_revenue_l365d         0.261276
bathrooms                       0.255766
review_scores_location          0.241287
review_scores_checkin           0.241287
review_scores_value             0.241287
review_scores_communication     0.241158
review_scores_cleanliness       0.241158
review_scores_accuracy          0.241158
reviews_per_month               0.241030

Missingness snapshot (reviews):
reviewer_name    0.00001
listing_id       0.00000
id               0.00000
date             0.00000
reviewer_id      0.00000
comments         0.00000
review_length    0.00000
'''

# %%
# ==============================
# LOAD CLEANED LISTINGS FIRST
# ==============================
listings = pd.read_csv("/Users/joshuachen/Projects/Airbnb-Investment-Management-Platform/data/processed/listings_clean.csv")


# =====================================================
# 1. DROP COLUMNS WE DO NOT NEED (based on missingness)
# =====================================================

drop_cols = [
    "bathrooms_text",
    "calendar_updated",
    "neighbourhood_group_cleansed",
    "neighborhood_overview",
    "host_about",
    "license"
]

listings = listings.drop(columns=[c for c in drop_cols if c in listings.columns])


# =====================================================
# 2. NUMERIC IMPUTATION WITH MISSING INDICATORS
# =====================================================

numeric_impute_cols = [
    "bathrooms",
    "review_scores_location",
    "review_scores_checkin",
    "review_scores_value",
    "review_scores_communication",
    "review_scores_cleanliness",
    "review_scores_accuracy",
    "reviews_per_month",
    "estimated_revenue_l365d"
]

for col in numeric_impute_cols:
    if col in listings.columns:

        # create missing indicator feature
        listings[f"{col}_missing"] = listings[col].isna().astype(int)

        # median imputation
        median_val = listings[col].median()
        listings[col] = listings[col].fillna(median_val)


# =====================================================
# 3. CATEGORICAL SAFETY IMPUTATION
# =====================================================

cat_cols = [
    "property_type",
    "room_type",
    "neighbourhood_cleansed"
]

for col in cat_cols:
    if col in listings.columns:
        listings[col] = listings[col].fillna("Unknown")


# =====================================================
# 4. SAVE IMPUTED DATASET
# =====================================================

listings.to_csv("/Users/joshuachen/Projects/Airbnb-Investment-Management-Platform/data/processed/listings_clean.csv", index=False)

print("Listings imputation complete.")
print(listings.isna().mean().sort_values(ascending=False).head(15))

'''
Listings imputation complete.
last_review                    0.241030
first_review                   0.241030
host_location                  0.225397
host_response_time             0.138775
host_response_rate             0.138775
host_acceptance_rate           0.102255
host_neighbourhood             0.042158
has_availability               0.033060
host_is_superhost              0.018708
description                    0.012686
review_scores_location         0.000000
review_scores_communication    0.000000
review_scores_checkin          0.000000
review_scores_cleanliness      0.000000
review_scores_accuracy         0.000000
'''
# %%
