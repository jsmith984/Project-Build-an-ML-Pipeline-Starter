import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "price"

# Columns that should not be used to predict price.
DROP_COLUMNS = [
    "id",
    "host_id",
    "host_name",
    "last_review",
]


def build_preprocessor(df, max_tfidf_features=5):
    """
    Build preprocessing transformations for Airbnb data.

    Numerical columns:
        Missing values are replaced with the median.

    Categorical columns:
        Missing values are replaced with the most frequent value.
        One-hot encoding is applied.

    Text column:
        TF-IDF is applied to the listing name.
    """

    excluded = [TARGET] + DROP_COLUMNS

    feature_columns = [
        col for col in df.columns
        if col not in excluded
    ]

    numerical_columns = [
        col for col in feature_columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_columns = [
        col for col in feature_columns
        if (
            col not in numerical_columns
            and col != "name"
        )
    ]

    transformers = []

    # Numerical preprocessing.
    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    # Categorical preprocessing.
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    # Text preprocessing.
    if "name" in feature_columns:
        transformers.append(
            (
                "title",
                TfidfVectorizer(
                    max_features=int(max_tfidf_features),
                    stop_words="english",
                    lowercase=True,
                    strip_accents="unicode",
                ),
                "name",
            )
        )

    if not transformers:
        raise ValueError("No usable features found in the dataset.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


def prepare_features(df):
    """
    Prepare the feature dataframe and target variable.
    """

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found."
        )

    X = df.drop(
        columns=[TARGET] + [
            col for col in DROP_COLUMNS
            if col in df.columns
        ],
        errors="ignore",
    ).copy()

    y = pd.to_numeric(df[TARGET], errors="coerce")

    # TF-IDF requires strings rather than missing values.
    if "name" in X.columns:
        X["name"] = X["name"].fillna("").astype(str)

    return X, y