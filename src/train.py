from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from data import (
    build_preprocessor,
    load_data,
    split_features_target,
)


DATA_FILE = "data/healthcare-dataset-stroke-data.csv"

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def build_random_forest(class_weight=None):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    class_weight=class_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_logistic_regression(class_weight=None):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def get_candidate_models():
    return {
        "Random Forest":
            build_random_forest(),

        "Class-Weighted Random Forest":
            build_random_forest(
                class_weight="balanced"
            ),

        "Logistic Regression":
            build_logistic_regression(),

        "Class-Weighted Logistic Regression":
            build_logistic_regression(
                class_weight="balanced"
            ),
    }


def cross_validate_models(models, X, y):
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
    }

    rows = []

    print("\n" + "=" * 70)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        rows.append(row)

        print(f"\n{name}")

        print(
            f"ROC-AUC:            "
            f"{row['roc_auc_mean']:.4f} "
            f"± {row['roc_auc_std']:.4f}"
        )

        print(
            f"PR-AUC:             "
            f"{row['pr_auc_mean']:.4f} "
            f"± {row['pr_auc_std']:.4f}"
        )

        print(
            f"Balanced Accuracy:  "
            f"{row['balanced_accuracy_mean']:.4f} "
            f"± {row['balanced_accuracy_std']:.4f}"
        )

        print(
            f"Stroke Recall:       "
            f"{row['recall_mean']:.4f} "
            f"± {row['recall_std']:.4f}"
        )

        print(
            f"Stroke Precision:    "
            f"{row['precision_mean']:.4f} "
            f"± {row['precision_std']:.4f}"
        )

        print(
            f"Stroke F1:           "
            f"{row['f1_mean']:.4f} "
            f"± {row['f1_std']:.4f}"
        )

    return pd.DataFrame(rows)


def evaluate_holdout(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                predictions,
            ),
        "precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
        "recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
        "f1":
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities,
            ),
        "pr_auc":
            average_precision_score(
                y_test,
                probabilities,
            ),
    }

    print("\n" + "=" * 70)
    print("FINAL HOLDOUT EVALUATION")
    print("=" * 70)

    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    return metrics


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data(DATA_FILE)
    X, y = split_features_target(data)

    # The holdout set is separated before any
    # model comparison takes place.
    X_dev, X_holdout, y_dev, y_holdout = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(f"Total samples:       {len(X)}")
    print(f"Development samples: {len(X_dev)}")
    print(f"Holdout samples:     {len(X_holdout)}")

    print("\nDevelopment class distribution:")
    print(y_dev.value_counts().sort_index())

    print("\nHoldout class distribution:")
    print(y_holdout.value_counts().sort_index())

    models = get_candidate_models()

    cv_results = cross_validate_models(
        models,
        X_dev,
        y_dev,
    )

    cv_path = OUTPUT_DIR / "cv_results.csv"

    cv_results.to_csv(
        cv_path,
        index=False,
    )

    # PR-AUC is used as the primary selection
    # metric because the positive class is rare.
    best_index = (
        cv_results["pr_auc_mean"].idxmax()
    )

    best_name = cv_results.loc[
        best_index,
        "model",
    ]

    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    print(
        cv_results[
            [
                "model",
                "roc_auc_mean",
                "pr_auc_mean",
                "balanced_accuracy_mean",
                "recall_mean",
                "precision_mean",
                "f1_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\nSelected by mean CV PR-AUC: "
        f"{best_name}"
    )

    final_model = models[best_name]

    # Train the selected model using the entire
    # development set.
    final_model.fit(
        X_dev,
        y_dev,
    )

    holdout_metrics = evaluate_holdout(
        final_model,
        X_holdout,
        y_holdout,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": best_name,
                **holdout_metrics,
            }
        ]
    )

    metrics_path = (
        OUTPUT_DIR
        / "holdout_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    model_path = (
        MODEL_DIR
        / "stroke_model.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(f"CV results:      {cv_path}")
    print(f"Holdout metrics: {metrics_path}")
    print(f"Final model:     {model_path}")


if __name__ == "__main__":
    main()