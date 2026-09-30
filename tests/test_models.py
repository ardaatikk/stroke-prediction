import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from train import (
    build_logistic_regression,
    build_random_forest,
    evaluate_holdout,
    get_candidate_models,
)


def create_training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Create a small representative dataset that can
    pass through the complete preprocessing and
    classification pipeline.
    """

    X = pd.DataFrame(
        {
            "age": [
                67.0,
                45.0,
                72.0,
                38.0,
                59.0,
                29.0,
                81.0,
                50.0,
            ],
            "avg_glucose_level": [
                228.69,
                85.04,
                190.20,
                92.10,
                160.00,
                75.30,
                210.50,
                110.40,
            ],
            "bmi": [
                36.6,
                28.1,
                31.5,
                24.8,
                30.2,
                22.5,
                33.1,
                27.6,
            ],
            "hypertension": [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
            ],
            "heart_disease": [
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
            ],
            "gender": [
                "Male",
                "Female",
                "Female",
                "Male",
                "Male",
                "Female",
                "Male",
                "Female",
            ],
            "ever_married": [
                "Yes",
                "Yes",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "Yes",
            ],
            "work_type": [
                "Private",
                "Private",
                "Self-employed",
                "Private",
                "Govt_job",
                "Private",
                "Self-employed",
                "Govt_job",
            ],
            "Residence_type": [
                "Urban",
                "Rural",
                "Urban",
                "Rural",
                "Urban",
                "Urban",
                "Rural",
                "Urban",
            ],
            "smoking_status": [
                "formerly smoked",
                "never smoked",
                "smokes",
                "never smoked",
                "formerly smoked",
                "Unknown",
                "smokes",
                "never smoked",
            ],
        }
    )

    y = pd.Series(
        [
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
        ],
        name="stroke",
    )

    return X, y


def test_logistic_regression_pipeline_structure() -> None:
    """
    Logistic regression should be wrapped together
    with preprocessing in a single sklearn Pipeline.
    """

    model = build_logistic_regression(
        class_weight="balanced"
    )

    assert isinstance(
        model,
        Pipeline,
    )

    assert list(
        model.named_steps
    ) == [
        "preprocessor",
        "classifier",
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    assert isinstance(
        classifier,
        LogisticRegression,
    )

    assert (
        classifier.class_weight
        == "balanced"
    )


def test_random_forest_pipeline_structure() -> None:
    """
    Random forest should also use the shared
    preprocessing pipeline.
    """

    model = build_random_forest(
        class_weight="balanced"
    )

    assert isinstance(
        model,
        Pipeline,
    )

    classifier = model.named_steps[
        "classifier"
    ]

    assert isinstance(
        classifier,
        RandomForestClassifier,
    )

    assert (
        classifier.class_weight
        == "balanced"
    )

    assert (
        classifier.n_estimators
        == 500
    )


def test_candidate_models_are_available() -> None:
    """
    Model comparison should expose all four intended
    candidate configurations.
    """

    models = get_candidate_models()

    assert set(
        models.keys()
    ) == {
        "Random Forest",
        "Class-Weighted Random Forest",
        "Logistic Regression",
        "Class-Weighted Logistic Regression",
    }


def test_logistic_regression_can_fit_and_predict() -> None:
    """
    The complete preprocessing + classifier pipeline
    should train and produce valid predictions.
    """

    X, y = create_training_data()

    model = build_logistic_regression(
        class_weight="balanced"
    )

    model.fit(
        X,
        y,
    )

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    assert predictions.shape == (
        len(X),
    )

    assert probabilities.shape == (
        len(X),
        2,
    )

    assert set(
        np.unique(predictions)
    ).issubset(
        {0, 1}
    )

    assert np.all(
        probabilities >= 0.0
    )

    assert np.all(
        probabilities <= 1.0
    )

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_evaluate_holdout_returns_expected_metrics() -> None:
    """
    Holdout evaluation should return the metrics used
    by the project for imbalanced classification.
    """

    X, y = create_training_data()

    model = build_logistic_regression(
        class_weight="balanced"
    )

    model.fit(
        X,
        y,
    )

    metrics = evaluate_holdout(
        model,
        X,
        y,
    )

    assert set(
        metrics.keys()
    ) == {
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }

    for value in metrics.values():
        assert 0.0 <= value <= 1.0