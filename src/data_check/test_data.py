import numpy as np
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


def test_proper_boundaries(data):

    assert data["latitude"].between(
        40.5,
        41.2,
    ).all()

    assert data["longitude"].between(
        -74.25,
        -73.50,
    ).all()


def test_unique_listing_ids(data):

    assert data["id"].is_unique, (
        "Duplicate listing IDs found."
    )


def test_no_missing_neighbourhoods(data):

    assert data["neighbourhood_group"].notna().all()


def test_similar_neigh_distrib(
    data,
    ref_data,
    kl_threshold,
):

    sample_counts = data[
        "neighbourhood_group"
    ].value_counts(normalize=True)

    reference_counts = ref_data[
        "neighbourhood_group"
    ].value_counts(normalize=True)

    categories = sorted(
        set(sample_counts.index)
        | set(reference_counts.index)
    )

    sample_probs = np.array([
        sample_counts.get(category, 0)
        for category in categories
    ])

    reference_probs = np.array([
        reference_counts.get(category, 0)
        for category in categories
    ])

    epsilon = 1e-10

    sample_probs = (
        sample_probs + epsilon
    ) / (
        sample_probs.sum()
        + epsilon * len(categories)
    )

    reference_probs = (
        reference_probs + epsilon
    ) / (
        reference_probs.sum()
        + epsilon * len(categories)
    )

    divergence = scipy.stats.entropy(
        sample_probs,
        reference_probs,
    )

    assert divergence <= kl_threshold, (
        f"KL divergence {divergence:.6f} "
        f"exceeds threshold {kl_threshold}"
    )


def test_row_count(data):

    assert 15000 < len(data) < 1000000


def test_price_range(data, min_price, max_price):

    assert data["price"].between(
        min_price,
        max_price,
    ).all()
