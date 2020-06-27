import numpy as np
from tqdm import tqdm
from sklearn import metrics, ensemble

from cardio import utils
from .model import Model


class IsolationForest(Model):
    def __init__(self, threshold_steps=40):
        self.threshold_steps = threshold_steps

    def create_model(self, random_state, contamination):
        """
        Create an Isolation Forest model with tuned hyperparameters.

        This model does not require (allow) class balancing.
        """
        return ensemble.IsolationForest(
            n_estimators=300,
            max_features=0.3,
            max_samples=1000,
            verbose=0,
            random_state=random_state,
            contamination=contamination,
        )

    def fit_model(self, model, train_data, *args, **kwargs):
        model.fit(train_data)

    def predict_model(self, model, test_data):
        preds = model.predict(test_data)
        preds = 1 - ((preds + 1) / 2)
        return preds

    def run_experiment(
        self,
        train,
        test,
        baseline=None,
        bootstrap=False,
        num_trials=10,
        first_random_seed=0,
        *args,
        **kwargs
    ):
        thresholds = np.linspace(0, 1, self.threshold_steps)
        tpr = []
        fpr = []
        precision = []
        recall = []

        metric_dict = {
            "roc": {"youden": -1, "thresh": None},
            "pr": {"f1": -1, "thresh": None},
            "calibration": {},
        }

        for t in tqdm(thresholds):
            cumulative_preds = np.zeros(test["data"].shape[0])
            for i in range(num_trials):
                rng = np.random.RandomState(first_random_seed + i)
                model = self.create_model(random_state=rng, contamination=t)
                self.fit_model(model, train["data"])
                pred_test = self.predict_model(model, test["data"])
                cumulative_preds = cumulative_preds + pred_test
            pred_test = np.round(cumulative_preds / num_trials)
            tp = np.sum(np.logical_and(pred_test == 1, test["labels"] == 1))
            fp = np.sum(np.logical_and(pred_test == 1, test["labels"] == 0))
            tn = np.sum(np.logical_and(pred_test == 0, test["labels"] == 0))
            fn = np.sum(np.logical_and(pred_test == 0, test["labels"] == 1))
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
            precision.append(tp / (tp + fp))
            recall.append(tp / (tp + fn))

            roc_youden = (1 - fpr[-1]) + tpr[-1] - 1

            if (precision[-1] * recall[-1]) == 0:
                f1 = 0
            else:
                f1 = 2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1])

            if f1 > metric_dict["pr"]["f1"]:
                metric_dict["pr"]["f1"] = f1
                metric_dict["pr"]["threshold"] = t
                metric_dict["pr"]["mortality_rate"] = (tp + fp) / pred_test.shape[0]

            if roc_youden > metric_dict["roc"]["youden"]:
                metric_dict["roc"]["youden"] = roc_youden
                metric_dict["roc"]["threshold"] = t
                metric_dict["roc"]["mortality_rate"] = (tp + fp) / pred_test.shape[0]

        metric_dict["roc"]["fpr"] = np.array(fpr)
        metric_dict["roc"]["tpr"] = np.array(tpr)
        metric_dict["roc"]["auc"] = metrics.auc(fpr, tpr, reorder=True)
        metric_dict["roc"]["auc_ci"] = utils.roc_auc_ci(
            metric_dict["roc"]["auc"],
            np.sum(test["labels"] == 1),
            np.sum(test["labels"] == 0),
        )
        metric_dict["roc"]["auc_pval"] = None  # TODO: Implement

        metric_dict["pr"]["precision"] = np.array(precision)
        metric_dict["pr"]["recall"] = np.array(recall)
        metric_dict["pr"]["auc"] = metrics.auc(recall, precision)
        metric_dict["pr"]["auc_ci"] = utils.pr_auc_ci(
            metric_dict["pr"]["auc"], np.sum(test["labels"] == 1)
        )
        metric_dict["pr"]["auc_pval"] = None  # TODO: Implement

        metric_dict["calibration"]["ece"] = None
        metric_dict["calibration"]["mce"] = None

        if bootstrap:
            new_dict = {}
            new_dict["holdout"] = metric_dict
            new_dict["bootstrap"] = {
                "roc": {
                    "auc": None,
                    "auc_ci": [None, None],
                    "auc_pval": None,
                    "youden": None,
                    "youden_ci": [None, None],
                    "youden_pval": None,
                },
                "pr": {
                    "auc": None,
                    "auc_ci": [None, None],
                    "auc_pval": None,
                    "f1": None,
                    "f1_ci": [None, None],
                    "f1_pval": None,
                },
                "calibration": {
                    "ece": None,
                    "ece_ci": [None, None],
                    "ece_pval": None,
                    "ece": None,
                    "mce": None,
                    "mce_ci": [None, None],
                    "mce_pval": None,
                },
            }
            metric_dict = new_dict

        metric_dict["outputs"] = None
        return metric_dict
