import pickle
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from cardio.utils import get_metrics, get_bootstrap_metrics


class Model:
    @abstractmethod
    def create_model(self, random_state):
        pass

    @abstractmethod
    def fit_model(self, model, train_data, train_labels):
        pass

    @abstractmethod
    def predict_model(self, model, test_data):
        pass

    @staticmethod
    def save_model(model, path):
        # Use the standard sklearn persitence: https://scikit-learn.org/stable/modules/model_persistence.html
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def run_experiment(
        self,
        train,
        test,
        baseline=None,
        bootstrap=False,
        num_trials=10,
        first_random_seed=0,
        export_path=None,
    ):
        if bootstrap:
            metric_fun = get_bootstrap_metrics
        else:
            metric_fun = get_metrics
        best_pr_auc = 0
        best_model = None
        cumulative_preds = np.zeros(test["data"].shape[0])
        for i in tqdm(range(num_trials)):
            rng = np.random.RandomState(first_random_seed + i)
            model = self.create_model(random_state=rng)
            self.fit_model(
                model, train_data=train["data"], train_labels=train["labels"]
            )
            pred_test = self.predict_model(model, test_data=test["data"])
            cumulative_preds = cumulative_preds + pred_test
            if export_path is not None:
                model_metrics = get_metrics(test["labels"], pred_test)
                pr_auc = model_metrics["pr"]["auc"]
                if pr_auc > best_pr_auc:
                    best_model = model
        if export_path is not None:
            self.save_model(best_model, export_path)
        pred_test = cumulative_preds / num_trials
        metrics = metric_fun(test["labels"], pred_test, baseline=baseline)
        metrics["outputs"] = pred_test
        return metrics
