#%%
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- Project paths ----------------
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").exists():
            return p
    return start.parent

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LISTINGS_PATH = DATA_DIR / "listings_sf.csv"
REVIEWS_PATH = DATA_DIR / "reviews_sf.csv"


# ---------------- Load ----------------
def load_raw(listings_path: Path, reviews_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    listings = pd.read_csv(listings_path)
    reviews = pd.read_csv(reviews_path, usecols=["listing_id", "id", "date", "comments"])
    return listings, reviews


# ---------------- Feature engineering ----------------
def build_listings_features(listings: pd.DataFrame) -> pd.DataFrame:
    df = listings.copy()

    df["price_clean"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )

    df["host_response_rate_num"] = (
        pd.to_numeric(df["host_response_rate"].astype(str).str.replace("%", "", regex=False), errors="coerce") / 100.0
    )
    df["host_acceptance_rate_num"] = (
        pd.to_numeric(df["host_acceptance_rate"].astype(str).str.replace("%", "", regex=False), errors="coerce") / 100.0
    )

    tf_mapping = {"t": 1, "f": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    for col in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]:
        df[f"{col}_bin"] = df[col].astype(str).str.lower().map(tf_mapping)

    df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
    df["last_scraped_dt"] = pd.to_datetime(df["last_scraped"], errors="coerce")
    df["host_tenure_days"] = (df["last_scraped_dt"] - df["host_since_dt"]).dt.days

    amenities_text = (
        df["amenities"]
        .fillna("[]")
        .astype(str)
        .str.strip()
        .str.strip("[]")
    )
    df["amenities_count"] = amenities_text.apply(
        lambda s: 0.0 if s.strip() == "" else float(len([item for item in s.split(",") if item.strip()]))
    )

    return df


def build_review_agg(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["comments_len"] = df["comments"].fillna("").astype(str).str.len()

    agg = (
        df.groupby("listing_id", as_index=False)
        .agg(
            review_count=("id", "count"),
            mean_comment_length=("comments_len", "mean"),
            first_review_date=("date", "min"),
            last_review_date=("date", "max"),
        )
    )
    agg["review_active_days"] = (agg["last_review_date"] - agg["first_review_date"]).dt.days
    return agg


# ---------------- Save ----------------
def save_outputs(listings_feat: pd.DataFrame, review_agg: pd.DataFrame, out_dir: Path) -> Path:
    listings_out = out_dir / "listings_clean.csv"
    reviews_out = out_dir / "reviews_clean.csv"

    listings_feat.to_csv(listings_out, index=False)
    review_agg.to_csv(reviews_out, index=False)

    merged = (
        listings_feat.merge(review_agg, left_on="id", right_on="listing_id", how="left")
        .drop(columns=["listing_id"])
    )
    merged_out = out_dir / "reviews_with_listings.csv"
    merged.to_csv(merged_out, index=False)

    return merged_out


def main() -> None:
    listings, reviews = load_raw(LISTINGS_PATH, REVIEWS_PATH)
    print("Listings shape:", listings.shape)
    print("Reviews shape:", reviews.shape)

    listings_feat = build_listings_features(listings)
    review_agg = build_review_agg(reviews)

    merged_path = save_outputs(listings_feat, review_agg, OUT_DIR)
    print("Saved merged file to:", merged_path)


if __name__ == "__main__":
    main()
