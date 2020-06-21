import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

from cardio import models
from cardio.utils import calibration_curve_quantile, ALL_FEATURES
from benchmark import load_split, TRAIN_SPLIT, TEST_SPLIT, NUM_TRIALS_PER_MODEL


if __name__ == "__main__":
    plt.ion()

    train = load_split(TRAIN_SPLIT)
    test = load_split(TEST_SPLIT, euroscore=True)

    results = {}

    # Evaluate baseline
    euroscore_preds = test.pop("euroscore")

    # Models for evaluation
    model_dict = {
        "random_forest": models.RandomForest(balanced=False),
        "svm_balanced": models.SVM(balanced=True),
        "gradient_boosting": models.GradientBoosting(balanced=False),
    }
    plot_models = {
        "euroscore": {
            "linestyle": "solid",
            "marker": "s",
            "first_color": "darkblue",
            "second_color": "royalblue",
        },
        "random_forest": {
            "linestyle": "solid",
            "marker": "s",
            "first_color": "darkgreen",
            "second_color": "limegreen",
        },
        "svm_balanced": {
            "linestyle": "solid",
            "marker": "s",
            "first_color": "darkred",
            "second_color": "indianred",
        },
        "gradient_boosting": {
            "linestyle": "solid",
            "marker": "s",
            "first_color": "darkorange",
            "second_color": "gold",
        },
    }

    # Calibration curves for the test population
    f, axs = plt.subplots()
    axs.plot([0, 1], [0, 1], "k:", label="perfectly_calibrated")
    pos_frac, mean_pred = calibration_curve_quantile(
        y_true=test["labels"], y_pred=euroscore_preds, n_bins=10
    )
    brier_score = brier_score_loss(test["labels"], euroscore_preds)
    axs.plot(mean_pred, pos_frac, "s-", label=f"euroscore - BD: {brier_score:.2f}")
    for name, model in model_dict.items():
        print(f"\n{name}")
        results[name] = model.run_experiment(
            train,
            test,
            baseline=euroscore_preds,
            bootstrap=False,
            num_trials=NUM_TRIALS_PER_MODEL,
        )
        if not isinstance(model, models.IsolationForest):
            brier_score = brier_score_loss(test["labels"], results[name]["outputs"])
            pos_frac, mean_pred = calibration_curve_quantile(
                y_true=test["labels"], y_pred=results[name]["outputs"], n_bins=10
            )
            axs.plot(
                mean_pred,
                pos_frac,
                "s-",
                color=plot_models[name]["first_color"],
                label=f"{name} - BD: {brier_score:.2f}",
            )
    axs.legend()
    axs.set_xlabel("Predicted Probability")
    axs.set_ylabel("Observed Probability")
    axs.minorticks_on()
    axs.grid(which="both")
    axs.set_aspect("equal")
    axs.set_title("All Population")

    # Calibration curves for risk factors
    risk_populations = {
        "Paciente: Edad": {
            "name": "Age",
            "groups": {
                "older than 75": lambda x: x > 75,
                # "younger or equal to 75": lambda x: x <= 75,
            },
        },
        "Fracción de Eyección (E)": {
            "name": "Ejection Fraction",
            "groups": {
                "less than 50": lambda x: x < 50,
                # "greater or equal to 50": lambda x: x >= 50,
            },
        },
        "Estado": {
            "name": "Admission",
            "groups": {
                "Non Elective": lambda x: x != 1,
                # "Rescue": lambda x: x == 4,
                # "Emergency": lambda x: x == 3,
                # "Urgent": lambda x: x == 2,
                # "Elective": lambda x: x == 1,
            },
        },
        "Arritmia": {
            "name": "Arrhythmia",
            "groups": {
                "Arrhythmia": lambda x: x == 1,
                # "No Arrhytmia": lambda x: x == 0,
            },
        },
        "Shock cardiogénico": {
            "name": "Cardiogenic Shock",
            "groups": {
                "Shock": lambda x: x == 1,
                # "No Shock": lambda x: x == 0
            },
        },
        "Resucitación": {
            "name": "Reanimation",
            "groups": {
                "Reanimation": lambda x: x == 1,
                # "No Reanimation": lambda x: x == 0,
            },
        },
        "Insuficiencia cardíaca": {
            "name": "Heart Failure",
            "groups": {
                "Failure": lambda x: x == 1,
                # "No Failure": lambda x: x == 0,
            },
        },
        "Insuficiencia renal - diálisis": {
            "name": "Dialysis",
            "groups": {
                "Dialysis": lambda x: x == 1,
                # "No Dialysis": lambda x: x == 0,
            },
        },
    }

    results["euroscore"] = {"outputs": euroscore_preds}

    # cmap = matplotlib.cm.get_cmap("RdBu")

    axes = []

    if len(risk_populations) <= 2:
        f, axs = plt.subplots(1, 2)
        axes = axs.ravel()
    else:
        done = False
        plots = 0
        while plots < len(risk_populations):
            f, axs = plt.subplots(1, 2)
            axes.append(axs.ravel())
            plots += 2
        axes = np.concatenate(axes, axis=0)

    plot_idx = 0
    y_true = test["labels"]
    for feature_name, info in risk_populations.items():
        index = test["features"].index(feature_name)
        display_name = None
        groups = info["groups"]
        num_groups = len(groups) - 1
        for model_name, plot_specs in plot_models.items():
            group_idx = 0
            for group_name, rule in groups.items():
                y_pred = results[model_name]["outputs"]
                include = rule(test["data"][:, index])
                if display_name is None:
                    display_name = f"{str(info['name']).upper()}: {group_name} - n = {include.sum()}"
                group_true = y_true[include]
                group_pred = y_pred[include]
                pos_frac, mean_pred = calibration_curve_quantile(group_true, group_pred)
                group_brier = brier_score_loss(group_true, group_pred)
                if group_idx == 0:
                    color = plot_specs["first_color"]
                elif group_idx == 1:
                    color = plot_specs["second_color"]
                else:
                    raise ValueError("More than 2 risk groups were provided")
                label = (
                    f"{str(model_name).upper()}: {group_name} - BS: {group_brier:.2f}"
                )
                axes[plot_idx].plot(
                    mean_pred,
                    pos_frac,
                    linestyle=plot_specs["linestyle"],
                    marker=plot_specs["marker"],
                    # color=cmap(float(group_idx) / num_groups),
                    color=color,
                    label=label,
                )
                group_idx += 1
        axes[plot_idx].plot([0, 1], [0, 1], "k:")
        axes[plot_idx].set_title(display_name)
        axes[plot_idx].minorticks_on()
        axes[plot_idx].grid(which="both")
        axes[plot_idx].legend()
        axes[plot_idx].set_aspect("equal")
        axes[plot_idx].set_xlabel("Predicted Probability")
        axes[plot_idx].set_ylabel("Observed Probability")
        plot_idx += 1
