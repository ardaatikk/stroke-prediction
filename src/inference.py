import argparse

import joblib
import pandas as pd


MODEL_FILE = "models/stroke_model.joblib"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run stroke-risk classification "
            "for a single input sample."
        )
    )

    parser.add_argument("--age", type=float, required=True)
    parser.add_argument(
        "--avg_glucose_level",
        type=float,
        required=True,
    )
    parser.add_argument("--bmi", type=float, required=True)

    parser.add_argument(
        "--hypertension",
        type=int,
        choices=[0, 1],
        required=True,
    )

    parser.add_argument(
        "--heart_disease",
        type=int,
        choices=[0, 1],
        required=True,
    )

    parser.add_argument("--gender", required=True)
    parser.add_argument("--ever_married", required=True)
    parser.add_argument("--work_type", required=True)
    parser.add_argument("--Residence_type", required=True)
    parser.add_argument("--smoking_status", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    model = joblib.load(MODEL_FILE)

    sample = pd.DataFrame(
        [
            {
                "age": args.age,
                "avg_glucose_level":
                    args.avg_glucose_level,
                "bmi": args.bmi,
                "hypertension":
                    args.hypertension,
                "heart_disease":
                    args.heart_disease,
                "gender": args.gender,
                "ever_married":
                    args.ever_married,
                "work_type": args.work_type,
                "Residence_type":
                    args.Residence_type,
                "smoking_status":
                    args.smoking_status,
            }
        ]
    )

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(
        sample
    )[0, 1]

    label = (
        "Stroke"
        if prediction == 1
        else "No Stroke"
    )

    print(f"Prediction: {label}")
    print(
        f"Model probability: "
        f"{probability:.4f}"
    )

    print(
        "\nNote: This model is an educational "
        "machine-learning project and is not "
        "intended for clinical diagnosis."
    )


if __name__ == "__main__":
    main()