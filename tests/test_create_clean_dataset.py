"""
Tests for create_clean_dataset.py
"""
import numpy as np
import pandas as pd
import pytest

# ── Import the helpers directly ───────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from create_clean_dataset import (
    clean_price,
    clean_percentage,
    parse_boolean,
    parse_bathrooms_text,
    count_amenities,
    engineer_features,
    merge_and_finalise,
)


# ── clean_price ───────────────────────────────────────────────────────────────

class TestCleanPrice:
    def test_strips_dollar_sign(self):
        s = pd.Series(["$150.00"])
        assert clean_price(s)[0] == 150.0

    def test_strips_comma_and_dollar(self):
        s = pd.Series(["$1,200.00"])
        assert clean_price(s)[0] == 1200.0

    def test_plain_number_string(self):
        s = pd.Series(["75"])
        assert clean_price(s)[0] == 75.0

    def test_invalid_returns_nan(self):
        s = pd.Series(["N/A"])
        assert pd.isna(clean_price(s)[0])

    def test_multiple_values(self):
        s = pd.Series(["$50.00", "$1,000.00", "N/A"])
        result = clean_price(s)
        assert result[0] == 50.0
        assert result[1] == 1000.0
        assert pd.isna(result[2])


# ── clean_percentage ──────────────────────────────────────────────────────────

class TestCleanPercentage:
    def test_strips_percent_sign(self):
        s = pd.Series(["95%"])
        assert clean_percentage(s)[0] == 95.0

    def test_plain_number(self):
        s = pd.Series(["100"])
        assert clean_percentage(s)[0] == 100.0

    def test_invalid_returns_nan(self):
        s = pd.Series(["N/A"])
        assert pd.isna(clean_percentage(s)[0])


# ── parse_boolean ─────────────────────────────────────────────────────────────

class TestParseBoolean:
    def test_t_maps_to_1(self):
        s = pd.Series(["t"])
        assert parse_boolean(s)[0] == 1

    def test_f_maps_to_0(self):
        s = pd.Series(["f"])
        assert parse_boolean(s)[0] == 0

    def test_true_bool(self):
        s = pd.Series([True])
        assert parse_boolean(s)[0] == 1

    def test_false_bool(self):
        s = pd.Series([False])
        assert parse_boolean(s)[0] == 0

    def test_unknown_returns_nan(self):
        s = pd.Series(["maybe"])
        assert pd.isna(parse_boolean(s)[0])


# ── parse_bathrooms_text ──────────────────────────────────────────────────────

class TestParseBathroomsText:
    def test_integer_bath(self):
        assert parse_bathrooms_text("1 bath") == 1.0

    def test_float_bath(self):
        assert parse_bathrooms_text("1.5 baths") == 1.5

    def test_half_bath(self):
        assert parse_bathrooms_text("Half-bath") == 0.5

    def test_nan_input(self):
        assert pd.isna(parse_bathrooms_text(np.nan))

    def test_no_number_returns_nan(self):
        assert pd.isna(parse_bathrooms_text("no bathrooms"))

    def test_zero_bath(self):
        assert parse_bathrooms_text("0 baths") == 0.0


# ── count_amenities ───────────────────────────────────────────────────────────

class TestCountAmenities:
    def test_counts_quoted_items(self):
        amenities = '["Wifi", "Kitchen", "Washer"]'
        assert count_amenities(amenities) == 3

    def test_empty_list_returns_zero(self):
        assert count_amenities("[]") == 0

    def test_nan_returns_zero(self):
        assert count_amenities(np.nan) == 0

    def test_single_item(self):
        assert count_amenities('["Wifi"]') == 1


# ── engineer_features ─────────────────────────────────────────────────────────

def _make_listings_df(n=5) -> pd.DataFrame:
    """Minimal listings DataFrame with all required columns."""
    return pd.DataFrame({
        "id":                    range(n),
        "last_scraped":          ["2024-01-01"] * n,
        "price":                 [f"${100 + i * 10}.00" for i in range(n)],
        "host_is_superhost":     ["t", "f", "t", "f", "t"],
        "host_response_rate":    ["100%", "90%", "80%", "70%", "60%"],
        "host_acceptance_rate":  ["95%", "85%", "75%", "65%", "55%"],
        "host_has_profile_pic":  ["t"] * n,
        "host_identity_verified":["t"] * n,
        "instant_bookable":      ["f"] * n,
        "has_availability":      ["t"] * n,
        "host_response_time":    ["within an hour"] * n,
        "host_since":            ["2020-01-01"] * n,
        "first_review":          ["2021-01-01"] * n,
        "last_review":           ["2023-06-01"] * n,
        "bathrooms_text":        ["1 bath", "2 baths", "Half-bath", "1.5 baths", "3 baths"],
        "amenities":             ['["Wifi", "Kitchen"]'] * n,
        "room_type":             ["Entire home/apt", "Private room", "Shared room",
                                  "Entire home/apt", "Private room"],
        "neighbourhood_cleansed":["Mission", "SoMa", "Castro", "Noe Valley", "Marina"],
        "property_type":         ["Entire rental unit"] * n,
        "latitude":              [37.75 + i * 0.01 for i in range(n)],
        "longitude":             [-122.4 + i * 0.01 for i in range(n)],
    })


class TestEngineerFeatures:
    def test_returns_dataframe(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_price_clean_column_exists(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert "price_clean" in result.columns

    def test_price_clean_values_are_numeric(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert pd.api.types.is_float_dtype(result["price_clean"])

    def test_superhost_parsed_to_numeric(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert set(result["host_is_superhost"].dropna().unique()).issubset({0, 1})

    def test_room_type_flags(self):
        df = _make_listings_df()
        result = engineer_features(df)
        # First row is "Entire home/apt"
        assert result.loc[0, "is_entire_home"] == 1
        assert result.loc[0, "is_private_room"] == 0
        assert result.loc[0, "is_shared_room"] == 0

    def test_amenity_count_positive(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert (result["amenity_count"] >= 0).all()

    def test_has_wifi_flag(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert "has_wifi" in result.columns
        assert result["has_wifi"].iloc[0] == 1

    def test_neighbourhood_top_column(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert "neighbourhood_top" in result.columns

    def test_property_type_simple_column(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert "property_type_simple" in result.columns

    def test_host_tenure_years_positive(self):
        df = _make_listings_df()
        result = engineer_features(df)
        assert (result["host_tenure_years"].dropna() >= 0).all()

    def test_response_time_mapped(self):
        df = _make_listings_df()
        result = engineer_features(df)
        # "within an hour" → 1
        assert result["host_response_time_num"].iloc[0] == 1

    def test_row_count_preserved(self):
        df = _make_listings_df(10)
        result = engineer_features(df)
        assert len(result) == 10


# ── merge_and_finalise ────────────────────────────────────────────────────────

class TestMergeAndFinalise:
    def test_merge_adds_review_columns(self):
        listings = _make_listings_df()
        listings = engineer_features(listings)
        review_agg = pd.DataFrame({
            "listing_id":             [0, 1, 2],
            "sentiment_polarity_mean":[0.5, 0.3, -0.1],
        })
        result = merge_and_finalise(listings, review_agg)
        assert "sentiment_polarity_mean" in result.columns

    def test_listings_without_reviews_kept(self):
        listings = _make_listings_df(5)
        listings = engineer_features(listings)
        review_agg = pd.DataFrame({
            "listing_id":             [0],
            "sentiment_polarity_mean":[0.5],
        })
        result = merge_and_finalise(listings, review_agg)
        # All 5 original listings should still be present (left join)
        assert len(result) == 5

    def test_index_column_added(self):
        listings = _make_listings_df()
        listings = engineer_features(listings)
        result = merge_and_finalise(listings, pd.DataFrame({"listing_id": []}))
        assert "_index" in result.columns
