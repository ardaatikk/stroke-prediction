from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    FEATURES,
    TARGET,
    build_preprocessor,
    load_data,
    split_features_target,
)


def create_sample_data() -> pd.DataFrame:
    """
    Create a small representative dataset for tests.
    """

    return pd.DataFrame(
        {
            "age": [67.0, 45.0, 52.0],
            "avg_glucose_level": [228.69, 85.04, 105.50],
            "bmi": [36.6, np.nan, 27.4],
            "hypertension": [0, 1, 0],
            "heart_disease": [1, 0, 0],
            "gender": ["Male", "Female", "Male"],
            "ever_married": ["Yes", "Yes", "No"],
            "work_type": [
                "Private",
                "Self-employed",
                "Private",
            ],
            "Residence_type": ["Urban", "Rural", "Urban"],
            "smoking_status": [
                "formerly smoked",
                "never smoked",
                "Unknown",
            ],
            "stroke": [1, 0, 0],
        }
    )


def test_load_data_reads_csv(
    tmp_path: Path,
) -> None:
    """
    load_data should read an existing CSV file.
    """

    data = create_sample_data()

    csv_path = tmp_path / "stroke.csv"

    data.to_csv(
        csv_path,
        index=False,
    )

    loaded = load_data(
        csv_path
    )

    pd.testing.assert_frame_equal(
        loaded,
        data,
    )


def test_load_data_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    load_data should reject a path that does not exist.
    """

    missing_path = (
        tmp_path
        / "missing.csv"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_data(
            missing_path
        )


def test_split_features_target() -> None:
    """
    Feature and target columns should be separated
    according to the project schema.
    """

    data = create_sample_data()

    X, y = split_features_target(
        data
    )

    assert list(X.columns) == FEATURES

    assert y.name == TARGET

    assert len(X) == len(data)

    assert len(y) == len(data)

    assert TARGET not in X.columns


def test_split_features_target_rejects_missing_target() -> None:
    """
    A dataset without the target column should fail.
    """

    data = create_sample_data().drop(
        columns=[TARGET]
    )

    with pytest.raises(
        ValueError,
        match="stroke",
    ):
        split_features_target(
            data
        )


def test_split_features_target_rejects_missing_feature() -> None:
    """
    A dataset missing a required model feature
    should fail.
    """

    data = create_sample_data().drop(
        columns=["age"]
    )

    with pytest.raises(
        ValueError,
        match="age",
    ):
        split_features_target(
            data
        )


def test_preprocessor_handles_missing_values() -> None:
    """
    The preprocessing pipeline should successfully
    transform data containing missing numeric values.
    """

    data = create_sample_data()

    X, _ = split_features_target(
        data
    )

    preprocessor = build_preprocessor()

    transformed = (
        preprocessor.fit_transform(
            X
        )
    )

    assert transformed.shape[0] == len(
        data
    )

    assert transformed.shape[1] > 0

    assert not np.isnan(
        transformed.astype(float)
    ).any()


def test_preprocessor_handles_unknown_category() -> None:
    """
    Categories unseen during fitting should not
    cause inference to fail.
    """

    training_data = create_sample_data()

    X_train, _ = split_features_target(
        training_data
    )

    preprocessor = build_preprocessor()

    preprocessor.fit(
        X_train
    )

    unseen_sample = X_train.iloc[
        [0]
    ].copy()

    unseen_sample.loc[
        :,
        "work_type",
    ] = "NeverSeenBefore"

    transformed = preprocessor.transform(
        unseen_sample
    )

    assert transformed.shape[0] == 1