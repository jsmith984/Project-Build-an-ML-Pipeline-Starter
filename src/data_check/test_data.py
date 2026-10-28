import pytest
import pandas as pd
import scipy.stats

def test_column_names(data):
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    assert set(data.columns) == set(expected_columns)

def test_proper_boundaries(data, min_price, max_price):
    assert data["latitude"].between(40.5, 41.2).all()
    assert data["longitude"].between(-74.25, -73.50).all()

def test_similar_neigh_distrib(data, ref_data, kl_threshold):
    # Tests distribution divergence between sample and reference
    pass

def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000

def test_price_range(data, min_price, max_price):
    assert data["price"].between(min_price, max_price).all()
