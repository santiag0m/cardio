import json

import numpy as np
import pandas as pd

from benchmark import load_split, TEST_SPLIT

OUTPUT_PATH = "data/outputs.json"

RISK_POPULATIONS = {
    "Paciente: Edad": {"name": "Age", "risk_function": lambda x: x > 75,},
    "Fracción de Eyección (E)": {
        "name": "Ejection Fraction",
        "risk_function": lambda x: x < 50,
    },
    "Estado": {"name": "Admission", "risk_function": lambda x: x != 1,},
    "Arritmia": {"name": "Arrhythmia", "risk_function": lambda x: x == 1,},
    "Shock cardiogénico": {
        "name": "Cardiogenic Shock",
        "risk_function": lambda x: x == 1,
    },
    "Resucitación": {"name": "Reanimation", "risk_function": lambda x: x == 1,},
    "Insuficiencia cardíaca": {
        "name": "Heart Failure",
        "risk_function": lambda x: x == 1,
    },
    "Insuficiencia renal - diálisis": {
        "name": "Dialysis",
        "risk_function": lambda x: x == 1,
    },
}


def calibration_metrics(model_scores, labels, at_risk_bool_index):
    high_scores = model_scores[at_risk_bool_index]
    high_labels = labels[at_risk_bool_index]
    low_scores = model_scores[np.logical_not(at_risk_bool_index)]
    low_labels = labels[np.logical_not(at_risk_bool_index)]

    high_pos_fraction = high_labels.mean()
    high_mean_predictive_value = high_scores.mean()
    high_calibration_error = np.abs(high_pos_fraction - high_mean_predictive_value)
    low_pos_fraction = low_labels.mean()
    low_mean_predictive_value = low_scores.mean()
    low_calibration_error = np.abs(low_pos_fraction - low_mean_predictive_value)

    return {
        "high_risk_n": at_risk_bool_index.sum(),
        "high_risk_positive_fraction": high_pos_fraction,
        "high_risk_mean_predicted_vakue": high_mean_predictive_value,
        "high_risk_calibration_error": high_calibration_error,
        "low_risk_n": len(at_risk_bool_index) - at_risk_bool_index.sum(),
        "low_risk_positive_fraction": low_pos_fraction,
        "low_risk_mean_predicted_value": low_mean_predictive_value,
        "low_risk_calibration_error": low_calibration_error,
    }


if __name__ == "__main__":
    with open(OUTPUT_PATH, "r") as f:
        results = json.load(f)

    test_data = load_split(TEST_SPLIT, euroscore=True)
    test_labels = test_data["labels"]

    # Check calibration in risk groups for selected models
    selected_models = ["euroscore", "gradient_boosting", "random_forest"]
    risk_populations = []
    for feature_name, properties in RISK_POPULATIONS.items():
        display_name = properties["name"]
        risk_function = properties["risk_function"]
        risk_feature = {}
        for model_name in selected_models:
            model_results = results[model_name]
            if "holdout" in model_results:
                roc_thresh = model_results["holdout"]["roc"]["threshold"]
                pr_thresh = model_results["holdout"]["pr"]["threshold"]
            else:
                roc_thresh = model_results["roc"]["threshold"]
                pr_thresh = model_results["pr"]["threshold"]
            model_scores = np.array(model_results["outputs"])
            feature_index = test_data["features"].index(feature_name)
            feature_values = test_data["data"][:, feature_index]
            model_high_risk = risk_function(feature_values)
            model_metrics = calibration_metrics(
                model_scores, test_labels, model_high_risk,
            )
            model_metrics["model_name"] = model_name
            model_metrics["group"] = feature_name.lower().replace(" ", "_")
            risk_populations.append(model_metrics)
    risk_populations = pd.DataFrame.from_records(risk_populations)
    risk_populations.to_csv("calibration_risk_populations.csv", index=False)

