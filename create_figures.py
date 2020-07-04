import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmark import load_split, TEST_SPLIT, OUTPUT_PATH

from confusion_matrix_pretty_print import plot_confusion_matrix_from_data


def get_thresholds(model_results):
    if "holdout" in model_results:
        roc_thresh = model_results["holdout"]["roc"]["threshold"]
        pr_thresh = model_results["holdout"]["pr"]["threshold"]
    else:
        roc_thresh = model_results["roc"]["threshold"]
        pr_thresh = model_results["pr"]["threshold"]
    return roc_thresh, pr_thresh


if __name__ == "__main__":
    plt.ion()

    with open(OUTPUT_PATH, "r") as f:
        results = json.load(f)

    test_data = load_split(TEST_SPLIT, euroscore=True)
    test_labels = test_data["labels"]

    euroscore = np.array(results["euroscore"]["outputs"])
    euroscore_roc_thresh, euroscore_pr_thresh = get_thresholds(results["euroscore"])
    gradient_boosting = np.array(results["gradient_boosting"]["outputs"])
    gradient_boosting_roc_thresh, gradient_boosting_pr_thresh = get_thresholds(
        results["gradient_boosting"]
    )

    fontsize = 18

    euroscore_roc, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=euroscore >= euroscore_roc_thresh,
        figsize=[10, 10],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    ax.set_title(
        f"Euroscore-II  ROC (Youden's J) - Threshold: {euroscore_roc_thresh * 100 :.02f}%",
        fontsize=fontsize + 2,
    )
    euroscore_roc.savefig("figures/euroscore_roc.png")

    euroscore_pr, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=euroscore >= euroscore_pr_thresh,
        figsize=[10, 10],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    ax.set_title(
        f"Euroscore-II  PR (F1) - Threshold: {euroscore_pr_thresh * 100 :.02f}%",
        fontsize=fontsize + 2,
    )
    euroscore_pr.savefig("figures/euroscore_pr.png")

    gradient_boosting_roc, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=gradient_boosting >= gradient_boosting_roc_thresh,
        figsize=[10, 10],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    ax.set_title(
        f"Gradient Boosting  ROC (Youden's J) - Threshold: {gradient_boosting_roc_thresh * 100 :.02f}%",
        fontsize=fontsize + 2,
    )
    gradient_boosting_roc.savefig("figures/gradient_boosting_roc.png")

    gradient_boosting_pr, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=gradient_boosting >= gradient_boosting_pr_thresh,
        figsize=[10, 10],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    ax.set_title(
        f"Gradient Boosting  PR (F1) - Threshold: {gradient_boosting_pr_thresh * 100 :.02f}%",
        fontsize=fontsize + 2,
    )
    gradient_boosting_pr.savefig("figures/gradient_boosting_pr.png")

    # Feature importance

    feature_importance = results["gradient_boosting"]["feature_importance"]
    indexes = list(feature_importance.keys())
    feat_names = [int(idx.replace("f", "")) for idx in indexes]
    feat_names = [test_data["features"][idx] for idx in feat_names]
    feat_values = [feature_importance[idx] for idx in indexes]
    ordered = np.argsort(feat_values)[::-1]

    feat_names = [feat_names[i] for i in ordered]
    feat_values = [feat_values[i] for i in ordered]

    fig, ax = plt.subplots()
    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, feat_values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Importance")
