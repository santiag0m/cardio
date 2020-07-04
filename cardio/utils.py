import numpy as np
import scipy.stats as scs
from sklearn import metrics

from .roc_comparison.compare_auc_delong_xu import delong_roc_test


ALL_FEATURES = [
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
]


def get_bootstrap_metrics(
    ground_truth, predicted, baseline=None, bootstrap_samples=10000
):
    roc_auc = []
    roc_youden = []
    pr_auc = []
    pr_f1 = []
    cal_ece = []
    cal_mce = []

    sample_size = ground_truth.shape[0]
    for _ in range(bootstrap_samples):
        idxs = np.random.choice(sample_size, sample_size, replace=True)
        sample_gt = ground_truth[idxs]
        sample_pred = predicted[idxs]
        sample_metrics = get_metrics(sample_gt, sample_pred)
        roc_auc.append(sample_metrics["roc"]["auc"])
        roc_youden.append(sample_metrics["roc"]["youden"])
        pr_auc.append(sample_metrics["pr"]["auc"])
        pr_f1.append(sample_metrics["pr"]["f1"])
        cal_ece.append(sample_metrics["calibration"]["ece"])
        cal_mce.append(sample_metrics["calibration"]["mce"])

    metric_dict = {}
    metric_dict["holdout"] = get_metrics(ground_truth, predicted, baseline=baseline)
    metric_dict["bootstrap"] = {
        "roc": {
            "auc": np.mean(roc_auc),
            "auc_ci": (np.quantile(roc_auc, 0.025), np.quantile(roc_auc, 0.975)),
            "auc_pval": None,
            "youden": np.mean(roc_youden),
            "youden_ci": (
                np.quantile(roc_youden, 0.025),
                np.quantile(roc_youden, 0.975),
            ),
            "youden_pval": None,
        },
        "pr": {
            "auc": np.mean(pr_auc),
            "auc_ci": (np.quantile(pr_auc, 0.025), np.quantile(pr_auc, 0.975)),
            "auc_pval": None,
            "f1": np.mean(pr_f1),
            "f1_ci": (np.quantile(pr_f1, 0.025), np.quantile(pr_f1, 0.975)),
            "f1_pval": None,
        },
        "calibration": {
            "ece": np.mean(cal_ece),
            "ece_ci": (np.quantile(cal_ece, 0.025), np.quantile(cal_ece, 0.975)),
            "ece_pval": None,
            "mce": np.mean(cal_mce),
            "mce_ci": (np.quantile(cal_mce, 0.025), np.quantile(cal_mce, 0.975)),
            "mce_pval": None,
        },
    }

    if baseline is not None:
        roc_auc = np.array(roc_auc)
        roc_youden = np.array(roc_youden)
        pr_auc = np.array(pr_auc)
        pr_f1 = np.array(pr_f1)
        cal_ece = np.array(cal_ece)
        cal_mce = np.array(cal_mce)

        metrics_baseline = get_metrics(ground_truth, baseline)

        roc_auc_baseline = metrics_baseline["roc"]["auc"]
        roc_youden_baseline = metrics_baseline["roc"]["youden"]
        pr_auc_baseline = metrics_baseline["pr"]["auc"]
        pr_f1_baseline = metrics_baseline["pr"]["f1"]
        cal_ece_baseline = metrics_baseline["calibration"]["ece"]
        cal_mce_baseline = metrics_baseline["calibration"]["mce"]

        roc_auc_diff = np.abs(metric_dict["holdout"]["roc"]["auc"] - roc_auc_baseline)
        roc_youden_diff = np.abs(
            metric_dict["holdout"]["roc"]["youden"] - roc_youden_baseline
        )
        pr_auc_diff = np.abs(metric_dict["holdout"]["pr"]["auc"] - pr_auc_baseline)
        pr_f1_diff = np.abs(metric_dict["holdout"]["pr"]["f1"] - pr_f1_baseline)
        cal_ece_diff = np.abs(
            metric_dict["holdout"]["calibration"]["ece"] - cal_ece_baseline
        )
        cal_mce_diff = np.abs(
            metric_dict["holdout"]["calibration"]["mce"] - cal_mce_baseline
        )

        metric_dict["bootstrap"]["roc"]["auc_pval"] = two_side_hypothesis_testing(
            roc_auc, roc_auc_baseline, roc_auc_diff
        )
        metric_dict["bootstrap"]["roc"]["youden_pval"] = two_side_hypothesis_testing(
            roc_youden, roc_youden_baseline, roc_youden_diff
        )
        metric_dict["bootstrap"]["pr"]["auc_pval"] = two_side_hypothesis_testing(
            pr_auc, pr_auc_baseline, pr_auc_diff
        )
        metric_dict["bootstrap"]["pr"]["f1_pval"] = two_side_hypothesis_testing(
            pr_f1, pr_f1_baseline, pr_f1_diff
        )
        metric_dict["bootstrap"]["calibration"][
            "ece_pval"
        ] = two_side_hypothesis_testing(cal_ece, cal_ece_baseline, cal_ece_diff)
        metric_dict["bootstrap"]["calibration"][
            "mce_pval"
        ] = two_side_hypothesis_testing(cal_mce, cal_mce_baseline, cal_mce_diff)

    return metric_dict


def two_side_hypothesis_testing(experiment, baseline, diff):
    # Hypothesis testing as described in: http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    sample_size = len(experiment)
    h0 = experiment - np.mean(experiment) + baseline
    outside = np.sum(h0 < (baseline - diff)) + np.sum(h0 > (baseline + diff))
    return outside / float(sample_size)


def get_metrics(ground_truth, predicted, baseline=None):
    """
    Calculate metrics for the ROC and PR curves.
    """
    metric_dict = {}
    metric_dict["roc"] = get_roc_metrics(ground_truth, predicted, baseline=baseline)
    metric_dict["pr"] = get_pr_metrics(ground_truth, predicted)
    metric_dict["calibration"] = get_calibration_metrics(ground_truth, predicted)
    return metric_dict


def get_roc_metrics(ground_truth, predicted, baseline=None):
    """
    Calculate metrics related to the Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, thresh_array = metrics.roc_curve(ground_truth, predicted, pos_label=1)
    j, thresh, mr = youden_optim(calc_youden(fpr, tpr), thresh_array, predicted)
    auc = metrics.auc(fpr, tpr)  # , reorder=True)
    auc_ci = roc_auc_ci(auc, np.sum(ground_truth == 1), np.sum(ground_truth == 0))
    metric_dict = {
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresh,
        "youden": j,
        "auc": auc,
        "auc_ci": auc_ci,
        "auc_pval": None,
        "mortality_rate": mr,
    }

    if baseline is not None:
        metric_dict["auc_pval"] = (
            10 ** delong_roc_test(ground_truth, baseline, predicted)[0][0]
        )

    return metric_dict


def get_pr_metrics(ground_truth, predicted, baseline=None):
    """
    Calculate metrics related to the Precision-Recall (PR) curve.
    """
    precision, recall, thresh_array = metrics.precision_recall_curve(
        ground_truth, predicted
    )
    precision = precision[:-1]
    recall = recall[:-1]
    f1, thresh, mr = foptim(calcf1(precision, recall), thresh_array, predicted)
    auc = metrics.auc(recall, precision)
    auc_ci = pr_auc_ci(auc, np.sum(ground_truth == 1))
    return {
        "precision": precision,
        "recall": recall,
        "threshold": thresh,
        "f1": f1,
        "auc": auc,
        "auc_ci": auc_ci,
        "auc_pval": None,
        "mortality_rate": mr,
    }


def get_calibration_metrics(ground_truth, predicted, baseline=None):
    """
    Calculate metrics related to the calibration/reliability curve as
    described in: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/
    """
    positive_fraction, mean_predicted_value, bin_counts = calibration_curve_quantile(
        ground_truth, predicted, n_bins=10, return_bin_counts=True
    )

    p_i = bin_counts / len(predicted)
    error = np.abs(positive_fraction - mean_predicted_value)
    ece = np.sum(p_i * error)
    mce = np.max(error)
    return {"ece": ece, "mce": mce}


def roc_auc_ci(auc, na, nn, alpha=0.05):
    """
    Confidence intervals for the Area Under the Curve for ROC.

    Based on the normality asumption proposed in:
    Hanley, James A., and Barbara J. McNeil. "The meaning and use of the area under a receiver operating characteristic (ROC) curve." Radiology 143.1 (1982): 29-36.

    And described in:
    https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf
    """
    # na: number abnormal
    z = scs.norm.ppf(1 - (alpha / 2))
    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    nom = auc * (1 - auc) + (na - 1) * (q1 - auc ** 2) + (nn - 1) * (q2 - auc ** 2)
    se = np.sqrt(nom / (na * nn))

    left = auc - z * se
    right = auc + z * se
    return left, right


def pr_auc_ci(auc, num_positive, alpha=0.05):
    """
    Confidence intervals for the Area Under the Curve for PR.

    Uses the logit interval described in:
    Boyd, Kendrick, Kevin H. Eng, and C. David Page. "Area under the precision-recall curve: point estimates and confidence intervals." Joint European conference on 
    machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2013.
    """
    z = scs.norm.ppf(1 - (alpha / 2))
    eta_hat = np.log(auc / (1 - auc))
    se = (num_positive * auc * (1 - auc)) ** (-0.5)

    left = np.exp(eta_hat - z * se)
    left = left / (1 + left)

    right = np.exp(eta_hat + z * se)
    right = right / (1 + right)
    return left, right


def calc_youden(fpr, tpr):
    sensitivity = tpr
    specificity = 1 - fpr
    youden = sensitivity + specificity - 1
    return youden


def youden_optim(youden, thresh, pred=None):
    """
    Find the maximum value for the Youden's J statistic, its corresponding threshold
    and the predicted mortality rate.
    """
    ind = np.argmax(youden)
    j = youden[ind]
    thresh = thresh[ind]
    if pred is not None:
        mr = np.sum(pred >= thresh) / pred.shape[0]
    else:
        mr = None
    return j, thresh, mr


def calcf1(precision, recall):
    "Calculate pointwise F1 score"
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    f1[np.isnan(f1)] = 0
    return f1


def foptim(f1, thresh, pred=None):
    """
    Find the maximum value for the F1 score, its corresponding threshold
    and the predicted mortality rate.
    """
    ind = np.argmax(f1)
    fmax = f1[ind]
    thresh = thresh[ind]
    if pred is not None:
        mr = np.sum(pred >= thresh) / pred.shape[0]
    else:
        mr = None
    return fmax, thresh, mr


def calibration_curve_quantile(y_true, y_pred, n_bins=5, return_bin_counts=False):
    assert len(y_pred) == len(y_true)
    indexes = np.argsort(y_pred)
    sorted_true = y_true[indexes]
    sorted_pred = y_pred[indexes]

    bin_size = max(len(y_pred) // n_bins, 1)

    positive_fraction = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        begin = i * bin_size
        if i == (n_bins - 1):
            end = len(y_pred)
        else:
            end = (i + 1) * bin_size
        positive_fraction[i] = sorted_true[begin:end].mean()
        mean_predicted_value[i] = sorted_pred[begin:end].mean()
        bin_counts[i] = end - begin

    if return_bin_counts:
        return positive_fraction, mean_predicted_value, bin_counts
    else:
        return positive_fraction, mean_predicted_value
