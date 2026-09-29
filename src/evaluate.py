from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data import (
    load_data,
    split_features_target,
)


DATA_FILE = "data/healthcare-dataset-stroke-data.csv"
MODEL_FILE = "models/stroke_model.joblib"

ASSET_DIR = Path("assets")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def save_confusion_matrix(y_true, predictions):
    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No Stroke",
            "Stroke",
        ],
    )

    display.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        ASSET_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_roc_curve(y_true, probabilities):
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
    )

    plt.title("ROC Curve")
    plt.tight_layout()

    plt.savefig(
        ASSET_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_precision_recall_curve(
    y_true,
    probabilities,
):
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
    )

    plt.title("Precision-Recall Curve")
    plt.tight_layout()

    plt.savefig(
        ASSET_DIR / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_feature_coefficients(model):
    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = classifier.coef_[0]

    order = np.argsort(
        np.abs(coefficients)
    )[-15:]

    selected_names = [
        feature_names[index]
        .replace("numeric__", "")
        .replace("binary__", "")
        .replace("categorical__", "")
        for index in order
    ]

    selected_coefficients = (
        coefficients[order]
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        selected_names,
        selected_coefficients,
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel("Logistic Regression Coefficient")
    plt.title(
        "Top Model Coefficients by Absolute Magnitude"
    )

    plt.tight_layout()

    plt.savefig(
        ASSET_DIR
        / "feature_coefficients.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    ASSET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data(DATA_FILE)
    X, y = split_features_target(data)

    _, X_holdout, _, y_holdout = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    model = joblib.load(MODEL_FILE)

    predictions = model.predict(
        X_holdout
    )

    probabilities = model.predict_proba(
        X_holdout
    )[:, 1]

    metrics = {
        "Balanced Accuracy":
            balanced_accuracy_score(
                y_holdout,
                predictions,
            ),
        "Precision":
            precision_score(
                y_holdout,
                predictions,
                zero_division=0,
            ),
        "Recall":
            recall_score(
                y_holdout,
                predictions,
                zero_division=0,
            ),
        "F1 Score":
            f1_score(
                y_holdout,
                predictions,
                zero_division=0,
            ),
        "ROC-AUC":
            roc_auc_score(
                y_holdout,
                probabilities,
            ),
        "PR-AUC":
            average_precision_score(
                y_holdout,
                probabilities,
            ),
    }

    print(
        f"Holdout samples: {len(X_holdout)}"
    )

    for name, value in metrics.items():
        print(f"{name:20s}: {value:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_holdout,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    metrics_path = (
        ASSET_DIR / "metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        for name, value in metrics.items():
            file.write(
                f"{name}: {value:.4f}\n"
            )

    save_confusion_matrix(
        y_holdout,
        predictions,
    )

    save_roc_curve(
        y_holdout,
        probabilities,
    )

    save_precision_recall_curve(
        y_holdout,
        probabilities,
    )

    save_feature_coefficients(model)

    print(
        f"\nEvaluation outputs saved to: "
        f"{ASSET_DIR}"
    )


if __name__ == "__main__":
    main()