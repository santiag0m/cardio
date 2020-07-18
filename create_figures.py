import json
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt

from cardio.utils import FEATURE_TRANSLATION
from benchmark import load_split, TEST_SPLIT, OUTPUT_PATH

from confusion_matrix_pretty_print import plot_confusion_matrix_from_data


def save_figure(fig, filename, dpi=1000, **kwargs):
    png_1 = BytesIO()
    fig.savefig(png_1, dpi=dpi, format="png", **kwargs)
    png_2 = Image.open(png_1)
    png_2.save(filename)
    png_1.close()


def get_thresholds(model_results):
    if "holdout" in model_results:
        roc_thresh = model_results["holdout"]["roc"]["threshold"]
        pr_thresh = model_results["holdout"]["pr"]["threshold"]
    else:
        roc_thresh = model_results["roc"]["threshold"]
        pr_thresh = model_results["pr"]["threshold"]
    return roc_thresh, pr_thresh


def plot_curve(labels, results, model_list, metric_name, ax=None):
    if ax is None:
        f, ax = plt.subplots()
    if metric_name == "pr":
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    elif metric_name == "roc":
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
    else:
        raise ValueError(f"Metric {metric_name} is not supported.")
    for model_name in model_list:
        preds = np.array(results[model_name]["outputs"])
        if metric_name == "pr":
            if model_name == "isolation_forest":
                y_axis = results[model_name]["holdout"]["pr"]["precision"]
                x_axis = results[model_name]["holdout"]["pr"]["recall"]
            else:
                precision, recall, _ = metrics.precision_recall_curve(labels, preds)
                y_axis = precision[:-1]
                x_axis = recall[:-1]
        elif metric_name == "roc":
            if model_name == "isolation_forest":
                y_axis = results[model_name]["holdout"]["roc"]["tpr"]
                x_axis = results[model_name]["holdout"]["roc"]["fpr"]
            else:
                fpr, tpr, thresh_array = metrics.roc_curve(labels, preds, pos_label=1)
                y_axis = tpr  # Sensitivity
                x_axis = fpr  # 1 - Specificity
        if model_name == "euroscore":
            alpha = 1
            lw = 1.5
            c = "b"
        elif "gradient_boosting" in model_name:
            alpha = 1
            lw = 1.5
            c = "r"
        else:
            alpha = 0.6
            lw = 1
            c = None
        if model_name in ["euroscore", "isolation_forest", "naive_bayes"]:
            model_name += "*"
        format_name = model_name.replace("_balanced", "")
        format_name = " ".join([word.capitalize() for word in format_name.split("_")])
        if format_name == "Euroscore*":
            format_name = "Euroscore II*"
        ax.plot(x_axis, y_axis, label=format_name, linewidth=lw, alpha=alpha, c=c)
    ax.axis("equal")


if __name__ == "__main__":

    with open(OUTPUT_PATH, "r") as f:
        results = json.load(f)

    test_data = load_split(TEST_SPLIT, euroscore=True)
    test_labels = test_data["labels"]

    # ROC and PR Curves
    _ = results.pop("isolation_forest", None)
    model_names = results.keys()
    balanced_models = [name for name in model_names if "balanced" in name]
    balanced_models = sorted(balanced_models)
    imbalanced_models = [name.replace("_balanced", "") for name in balanced_models]
    assert len(imbalanced_models) == len(
        list(set(imbalanced_models) & set(model_names))
    ), "Not all balanced models have an 'imbalanced' counterpart"
    remaining_models = list(
        set(model_names) - (set(imbalanced_models) | set(balanced_models))
    )
    remaining_models = sorted(remaining_models)
    balanced_models = remaining_models + balanced_models
    imbalanced_models = remaining_models + imbalanced_models

    # Plot ROC balanced and imbalanced
    f_imb, axs_imb = plt.subplots(1, 2)
    plot_curve(
        labels=test_labels,
        results=results,
        model_list=imbalanced_models,
        metric_name="roc",
        ax=axs_imb[0],
    )
    axs_imb[0].grid()
    axs_imb[0].set_title("ROC")
    plot_curve(
        labels=test_labels,
        results=results,
        model_list=imbalanced_models,
        metric_name="pr",
        ax=axs_imb[1],
    )
    axs_imb[1].grid()
    axs_imb[1].set_title("PR")
    title = f_imb.suptitle("Imbalanced")
    handles, labels = axs_imb[0].get_legend_handles_labels()
    f_imb.set_size_inches(7.3, 3.5)
    lgd = f_imb.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0, -0.3, 1, 1),
        bbox_transform=f_imb.transFigure,
        fancybox=True,
    )
    save_figure(
        f_imb,
        "figures/imbalanced_curves.tiff",
        bbox_extra_artists=(lgd, title),
        bbox_inches="tight",
    )
    # Plot ROC balanced and imbalanced
    f_bal, axs_bal = plt.subplots(1, 2)
    plot_curve(
        labels=test_labels,
        results=results,
        model_list=balanced_models,
        metric_name="roc",
        ax=axs_bal[0],
    )
    axs_bal[0].grid()
    axs_bal[0].set_title("ROC")
    plot_curve(
        labels=test_labels,
        results=results,
        model_list=balanced_models,
        metric_name="pr",
        ax=axs_bal[1],
    )
    axs_bal[1].grid()
    axs_bal[1].set_title("PR")
    title = f_bal.suptitle("Balanced")
    handles, labels = axs_bal[0].get_legend_handles_labels()
    f_bal.set_size_inches(7.3, 3.5)
    lgd = f_bal.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0, -0.3, 1, 1),
        bbox_transform=f_bal.transFigure,
        fancybox=True,
    )
    save_figure(
        f_bal,
        "figures/balanced_curves.tiff",
        bbox_extra_artists=(lgd, title),
        bbox_inches="tight",
    )

    plt.close("all")

    # Feature importance

    feature_importance = results["gradient_boosting"]["feature_importance"]
    indexes = list(feature_importance.keys())
    feat_names = [int(idx.replace("f", "")) for idx in indexes]
    feat_names = [test_data["features"][name] for name in feat_names]
    feat_names = [FEATURE_TRANSLATION[name] for name in feat_names]
    feat_values = [feature_importance[idx] for idx in indexes]
    ordered = np.argsort(feat_values)[::-1]

    feat_names = [feat_names[i] for i in ordered]
    feat_values = [feat_values[i] for i in ordered]

    f_imp, ax = plt.subplots(figsize=(7.3, 3.5))
    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, feat_values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Importance (Split Gain)")

    save_figure(f_imp, "figures/feature_importance.tiff", bbox_inches="tight")

    plt.close("all")

    # Create Confusion Matrices

    euroscore = np.array(results["euroscore"]["outputs"])
    euroscore_roc_thresh, euroscore_pr_thresh = get_thresholds(results["euroscore"])
    gradient_boosting = np.array(results["gradient_boosting"]["outputs"])
    gradient_boosting_roc_thresh, gradient_boosting_pr_thresh = get_thresholds(
        results["gradient_boosting"]
    )

    fontsize = 8

    euroscore_roc, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=euroscore >= euroscore_roc_thresh,
        figsize=[3.5, 3.5],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    title = ax.set_title(
        f"Euroscore II  ROC (Youden's J)\nThreshold: {euroscore_roc_thresh * 100 :.02f}%",
        fontsize=fontsize + 1,
    )
    save_figure(
        euroscore_roc,
        "figures/euroscore_roc.tiff",
        bbox_extra_artists=(title,),
        bbox_inches="tight",
    )

    euroscore_pr, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=euroscore >= euroscore_pr_thresh,
        figsize=[3.5, 3.5],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    title = ax.set_title(
        f"Euroscore II  PR (F1)\nThreshold: {euroscore_pr_thresh * 100 :.02f}%",
        fontsize=fontsize + 1,
    )
    save_figure(
        euroscore_pr,
        "figures/euroscore_pr.tiff",
        bbox_extra_artists=(title,),
        bbox_inches="tight",
    )

    gradient_boosting_roc, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=gradient_boosting >= gradient_boosting_roc_thresh,
        figsize=[3.5, 3.5],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    title = ax.set_title(
        f"Gradient Boosting  ROC (Youden's J)\nThreshold: {gradient_boosting_roc_thresh * 100 :.02f}%",
        fontsize=fontsize + 1,
    )
    save_figure(
        gradient_boosting_roc,
        "figures/gradient_boosting_roc.tiff",
        bbox_extra_artists=(title,),
        bbox_inches="tight",
    )

    gradient_boosting_pr, ax = plot_confusion_matrix_from_data(
        y_test=test_labels,
        predictions=gradient_boosting >= gradient_boosting_pr_thresh,
        figsize=[3.5, 3.5],
        fz=fontsize,
        columns=["Alive", "Dead"],
    )
    title = ax.set_title(
        f"Gradient Boosting  PR (F1)\nThreshold: {gradient_boosting_pr_thresh * 100 :.02f}%",
        fontsize=fontsize + 1,
    )
    save_figure(
        gradient_boosting_pr,
        "figures/gradient_boosting_pr.tiff",
        bbox_extra_artists=(title,),
        bbox_inches="tight",
    )

    plt.close("all")

