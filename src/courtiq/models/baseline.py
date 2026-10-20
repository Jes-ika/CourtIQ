"""Scikit-learn baseline model pipelines with preprocessing for missing and categorical values."""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_baseline_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build an end-to-end preprocessing and logistic regression pipeline.

    - Numeric features: Median imputation + Standard scaling.
    - Categorical features: Most-frequent imputation + One-hot encoding (handle_unknown='ignore').
    - Classifier: L2-regularized Logistic Regression.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return clf
