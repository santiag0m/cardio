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

    def run_experiment(
        self,
        train,
        test,
        baseline=None,
        bootstrap=False,
        num_trials=10,
        first_random_seed=0,
    ):
        if bootstrap:
            metric_fun = get_bootstrap_metrics
        else:
            metric_fun = get_metrics
        cumulative_preds = np.zeros(test["data"].shape[0])
        for i in tqdm(range(num_trials)):
            rng = np.random.RandomState(first_random_seed + i)
            model = self.create_model(random_state=rng)
            self.fit_model(
                model, train_data=train["data"], train_labels=train["labels"]
            )
            pred_test = self.predict_model(model, test_data=test["data"])
            cumulative_preds = cumulative_preds + pred_test
        pred_test = cumulative_preds / num_trials
        metrics = metric_fun(test["labels"], pred_test, baseline=baseline)
        metrics["outputs"] = pred_test
        return metrics
