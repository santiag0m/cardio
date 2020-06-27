import os

import pandas as pd
from tqdm import tqdm

from sklearn import preprocessing

from cardio import models
from cardio.utils import get_bootstrap_metrics, get_metrics


NUM_TRIALS_PER_MODEL = 10

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

SPLIT_SOURCE = os.path.join("data", "processedDataJune2020.xlsx")
EUROSCORE_SOURCE = os.path.join("data", "cardiodataMay2020.xlsx")

EXPORT_PATH = "benchmark_results_final_variables.csv"

USE_BOOTSTRAP = True

FEATURES = [
    "Paciente: Edad",
    "Paciente: Sexo_Femenino",
    "Indice de masa corporal",
    "Hipertensión",
    "Diabetes",
    "Enfermedad pulmonar crónica",
    "Enfermedad arterial periférica",
    "Enfermedad cerebro vascular",
    "Insuficiencia cardíaca",
    "Insuficiencia renal - diálisis",
    "Fracción de Eyección (E)",
    "Ultimo hematocrito",
    "Ultimo nivel de creatinina",
    "Endocarditis infecciosa",
    "Resucitación",
    "Shock cardiogénico",
    "Arritmia",
    "Número de vasos coronarios enfermos",
    "Insuficiencia aórtica (E)",
    "Insuficiencia mitral (E)",
    "Insuficiencia tricuspídea (E)",
    "Estado",
    "Peso del procedimiento - procedimiento aislado no CABG",
    "Peso del procedimiento - dos procedimientos",
    "Peso del procedimiento - tres o más procedimientos",
]

OUTCOME = "Muerte 30 días después de la cirugía"  # Death 30 days after surgery


def load_split(split_name, euroscore=False):
    output = {}
    # Load data splits
    df = pd.read_excel(SPLIT_SOURCE, sheet_name=split_name, index_col=0)
    output["data"] = df[FEATURES].values
    output["labels"] = df[OUTCOME].values
    output["features"] = FEATURES
    output["outcome"] = OUTCOME
    if euroscore:
        # Load baseline results
        baseline_df = pd.read_excel(EUROSCORE_SOURCE, sheet_name="Datos")
        baseline_df = baseline_df.set_index("Historia Clínica")
        baseline_df = baseline_df.loc[list(df.index), :]
        baseline = baseline_df["EUROSCORE"] / 100
        output["euroscore"] = baseline.values
    return output


def export_results(results):
    records = []
    for model_name, metrics in results.items():
        record = {}
        if "bootstrap" in metrics:
            records.append(
                {
                    "Model": model_name,
                    # ROC
                    "ROC AUC": metrics["holdout"]["roc"]["auc"],
                    "ROC AUC 5 (Normal)": metrics["holdout"]["roc"]["auc_ci"][0],
                    "ROC AUC 95 (Normal)": metrics["holdout"]["roc"]["auc_ci"][1],
                    "ROC AUC p-value (DeLong)": metrics["holdout"]["roc"]["auc_pval"],
                    "ROC AUC 5 (Bootstrap)": metrics["bootstrap"]["roc"]["auc_ci"][0],
                    "ROC AUC 95 (Bootstrap)": metrics["bootstrap"]["roc"]["auc_ci"][1],
                    "ROC AUC p-value (Bootstrap)": metrics["bootstrap"]["roc"][
                        "auc_pval"
                    ],
                    "ROC Youden's J": metrics["holdout"]["roc"]["youden"],
                    "ROC Youden's J 5 (Bootstrap) ": metrics["holdout"]["roc"][
                        "youden_c1"
                    ][0],
                    "ROC Youden's J 95 (Bootstrap) ": metrics["holdout"]["roc"][
                        "youden_c1"
                    ][1],
                    "ROC Youden's J p-value (Bootstrap) ": metrics["holdout"]["roc"][
                        "youden_pval"
                    ],
                    "ROC Threshold": metrics["holdout"]["roc"]["threshold"],
                    "ROC Mortality Rate": metrics["holdout"]["roc"]["mortality_rate"],
                    # PR
                    "PR AUC": metrics["holdout"]["pr"]["auc"],
                    "PR AUC 5 (Logit)": metrics["holdout"]["pr"]["auc_ci"][0],
                    "PR AUC 95 (Logit)": metrics["holdout"]["pr"]["auc_ci"][1],
                    "PR AUC 5 (Bootstrap)": metrics["bootstrap"]["pr"]["auc_ci"][0],
                    "PR AUC 95 (Bootstrap)": metrics["bootstrap"]["pr"]["auc_ci"][1],
                    "PR AUC p-value (Bootstrap)": metrics["bootstrap"]["pr"][
                        "auc_pval"
                    ],
                    "PR F1": metrics["holdout"]["pr"]["f1"],
                    "PR F1 5 (Bootstrap) ": metrics["holdout"]["pr"]["f1_c1"][0],
                    "PR F1 95 (Bootstrap) ": metrics["holdout"]["pr"]["f1_c1"][1],
                    "PR F1 p-value (Bootstrap) ": metrics["holdout"]["pr"]["f1_pval"],
                    "PR Threshold": metrics["holdout"]["pr"]["threshold"],
                    "PR Mortality Rate": metrics["holdout"]["pr"]["mortality_rate"],
                    # CALIBRATION
                    "ECE": metrics["holdout"]["calibration"]["ece"],
                    "ECE 5 (Bootstrap)": metrics["holdout"]["calibration"]["ece_ci"][0],
                    "ECE 95 (Bootstrap)": metrics["holdout"]["calibration"]["ece_ci"][
                        1
                    ],
                    "ECE p-value (Bootstrap)": metrics["holdout"]["calibration"][
                        "ece_pval"
                    ],
                    "MCE": metrics["holdout"]["calibration"]["mce"],
                    "MCE 5 (Bootstrap)": metrics["holdout"]["calibration"]["mce_ci"][0],
                    "MCE 95 (Bootstrap)": metrics["holdout"]["calibration"]["mce_ci"][
                        1
                    ],
                    "MCE p-value (Bootstrap)": metrics["holdout"]["calibration"][
                        "mce_pval"
                    ],
                }
            )
        else:
            records.append(
                {
                    "Model": model_name,
                    "ROC AUC": metrics["roc"]["auc"],
                    "ROC AUC 5 (Normal)": metrics["roc"]["auc_ci"][0],
                    "ROC AUC 95 (Normal)": metrics["roc"]["auc_ci"][1],
                    "ROC AUC p-value (DeLong)": metrics["roc"]["auc_pval"],
                    "ROC Youden's J": metrics["roc"]["youden"],
                    "ROC Threshold": metrics["roc"]["threshold"],
                    "ROC Mortality Rate": metrics["roc"]["mortality_rate"],
                    "PR AUC": metrics["pr"]["auc"],
                    "PR AUC 5 (Logit)": metrics["pr"]["auc_ci"][0],
                    "PR AUC 95 (Logit)": metrics["pr"]["auc_ci"][1],
                    # "PR AUC p-value": metrics["pr"]["auc_pval"],
                    "PR F1": metrics["pr"]["f1"],
                    "PR Threshold": metrics["pr"]["threshold"],
                    "PR Mortality Rate": metrics["pr"]["mortality_rate"],
                    "ECE": metrics["calibration"]["ece"],
                    "MCE": metrics["calibration"]["mce"],
                }
            )
    records = pd.DataFrame.from_records(records)
    records.to_csv(EXPORT_PATH, index=False)


if __name__ == "__main__":
    train = load_split(TRAIN_SPLIT)
    test = load_split(TEST_SPLIT, euroscore=True)

    results = {}

    # Evaluate baseline
    euroscore_preds = test.pop("euroscore")
    if USE_BOOTSTRAP:
        euroscore = get_bootstrap_metrics(
            ground_truth=test["labels"], predicted=euroscore_preds
        )
    else:
        euroscore = get_metrics(ground_truth=test["labels"], predicted=euroscore_preds)
    euroscore["outputs"] = euroscore_preds

    results["euroscore"] = euroscore

    # Evaluate models
    model_dict = {
        "isolation_forest": models.IsolationForest(),
        "logistic_regression": models.LogisticRegression(
            balanced=False, scaler="minmax"
        ),
        "logistic_regression_balanced": models.LogisticRegression(
            balanced=True, scaler="minmax"
        ),
        "support_vector_machine": models.SVM(balanced=False, scaler="minmax"),
        "support_vector_machine_balanced": models.SVM(balanced=True, scaler="minmax"),
        "random_forest": models.RandomForest(balanced=False),
        "random_forest_balanced": models.RandomForest(balanced=True),
        "multi_layer_perceptron": models.MLP(balanced=False),
        "multi_layer_perceptron_balanced": models.MLP(balanced=True),
        "naive_bayes": models.NaiveBayes(),
        "gradient_boosting": models.GradientBoosting(balanced=False),
        "gradient_boosting_balanced": models.GradientBoosting(balanced=True),
    }

    for name, model in model_dict.items():
        print(f"\n{name}")
        results[name] = model.run_experiment(
            train,
            test,
            baseline=euroscore_preds,
            bootstrap=USE_BOOTSTRAP,
            num_trials=NUM_TRIALS_PER_MODEL,
        )
        if USE_BOOTSTRAP:
            roc_auc = results[name]["holdout"]["roc"]["auc"]
            pr_auc = results[name]["holdout"]["pr"]["auc"]
        else:
            roc_auc = results[name]["roc"]["auc"]
            pr_auc = results[name]["pr"]["auc"]
        print(f"ROC AUC: {roc_auc:.3f} - PR AUC: {pr_auc:.3f}")

    export_results(results)
