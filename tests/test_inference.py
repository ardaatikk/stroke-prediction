import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


import inference
from train import build_logistic_regression


def create_training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Create a small representative dataset for
    inference tests.
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


def create_arguments() -> SimpleNamespace:
    """
    Create one representative CLI input sample.
    """

    return SimpleNamespace(
        age=67.0,
        avg_glucose_level=228.69,
        bmi=36.6,
        hypertension=1,
        heart_disease=1,
        gender="Male",
        ever_married="Yes",
        work_type="Private",
        Residence_type="Urban",
        smoking_status="formerly smoked",
    )


def test_saved_model_can_be_loaded_and_predict(
    tmp_path: Path,
) -> None:
    """
    A trained pipeline should survive joblib
    serialization and still support inference.
    """

    X, y = create_training_data()

    model = build_logistic_regression(
        class_weight="balanced"
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        tmp_path
        / "stroke_model.joblib"
    )

    inference.joblib.dump(
        model,
        model_path,
    )

    loaded_model = inference.joblib.load(
        model_path
    )

    sample = X.iloc[
        [0]
    ].copy()

    prediction = loaded_model.predict(
        sample
    )

    probabilities = (
        loaded_model.predict_proba(
            sample
        )
    )

    assert prediction.shape == (1,)
    assert probabilities.shape == (1, 2)

    assert int(
        prediction[0]
    ) in (0, 1)

    assert (
        0.0
        <= float(
            probabilities[0, 1]
        )
        <= 1.0
    )


def test_main_runs_single_sample_inference(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """
    The CLI inference path should load a model,
    process one raw sample, and print a result.
    """

    X, y = create_training_data()

    model = build_logistic_regression(
        class_weight="balanced"
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        tmp_path
        / "stroke_model.joblib"
    )

    inference.joblib.dump(
        model,
        model_path,
    )

    monkeypatch.setattr(
        inference,
        "MODEL_FILE",
        str(model_path),
    )

    monkeypatch.setattr(
        inference,
        "parse_args",
        create_arguments,
    )

    inference.main()

    captured = capsys.readouterr()

    assert "Prediction:" in captured.out

    assert (
        "Model probability:"
        in captured.out
    )

    assert (
        "not intended for clinical diagnosis"
        in captured.out
    )


def test_main_passes_expected_features_to_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    CLI inference should construct a sample with
    exactly the feature schema expected by the model.
    """

    X, y = create_training_data()

    model = build_logistic_regression(
        class_weight="balanced"
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        tmp_path
        / "stroke_model.joblib"
    )

    inference.joblib.dump(
        model,
        model_path,
    )

    monkeypatch.setattr(
        inference,
        "MODEL_FILE",
        str(model_path),
    )

    monkeypatch.setattr(
        inference,
        "parse_args",
        create_arguments,
    )

    inference.main()

    loaded_model = inference.joblib.load(
        model_path
    )

    assert list(
        loaded_model.feature_names_in_
    ) == [
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
