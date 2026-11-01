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


def test_proper_boundaries(data, min_price, max_price):
    assert data["latitude"].between(40.5, 41.2).all()
    assert data["longitude"].between(-74.25, -73.50).all()


def test_similar_neigh_distrib(data, ref_data, kl_threshold):
    sample_counts = data["neighbourhood_group"].value_counts(
        normalize=True
    )

    reference_counts = ref_data["neighbourhood_group"].value_counts(
        normalize=True
    )

    categories = sorted(
        set(sample_counts.index) | set(reference_counts.index)
    )

    sample_probs = np.array(
        [sample_counts.get(category, 0) for category in categories],
        dtype=float,
    )

    reference_probs = np.array(
        [reference_counts.get(category, 0) for category in categories],
        dtype=float,
    )

    # Smooth zero-frequency categories to avoid undefined KL values.
    epsilon = 1e-10

    sample_probs = (sample_probs + epsilon) / (
        sample_probs.sum() + epsilon * len(categories)
    )

    reference_probs = (reference_probs + epsilon) / (
        reference_probs.sum() + epsilon * len(categories)
    )

    kl_divergence = scipy.stats.entropy(
        sample_probs,
        reference_probs,
    )

    assert kl_divergence <= kl_threshold, (
        f"Neighborhood KL divergence {kl_divergence:.6f} "
        f"exceeds threshold {kl_threshold}"
    )


def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    assert data["price"].between(min_price, max_price).all()
