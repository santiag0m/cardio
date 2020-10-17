import shap
import numpy as np
import xgboost as xgb

from .model import Model


class GradientBoosting(Model):
    def __init__(self, balanced=False):
        self.balanced = balanced
        self.feature_importance = None
        self.shap_values = None
        self.num_trials = 0

    def create_model(self, random_state):
        seed_value = random_state.get_state()[1][0]
        param = {
            "max_depth": 2,
            "eta": 1,
            "lambda": 0.25,
            "subsample": 0.5,
            "nthread": 6,
            "eval_metric": ["auc", "aucpr"],
            "objective": "binary:logistic",
            "seed": seed_value,
        }
        num_round = 10
        return {
            "model": None,
            "param": param,
            "num_round": num_round,
        }

    def fit_model(self, model, train_data, train_labels):
        if self.balanced:
            # Do inverse weighting
            positives = train_labels.sum()
            negatives = train_labels.shape[0] - positives
            model["param"]["scale_pos_weight"] = negatives / float(positives)
        train_mat = xgb.DMatrix(data=train_data, label=train_labels)
        bst = xgb.train(
            params=model["param"],
            dtrain=train_mat,
            num_boost_round=model["num_round"],
            verbose_eval=True,
        )
        model["model"] = bst

        # Calculate Feature Importance Once Per Experiment
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(train_data)
        if self.feature_importance is None:
            self.feature_importance = bst.get_score(importance_type="gain")
        if self.shap_values is None:
            self.shap_values = shap_values
        else:
            # SHAP Value approximated for ensembles
            # https://github.com/slundberg/shap/issues/112
            self.shap_values = ((self.shap_values * self.num_trials) + shap_values) / (
                self.num_trials + 1
            )

        self.num_trials += 1

    def predict_model(self, model, test_data):
        test_mat = xgb.DMatrix(data=test_data)
        preds = model["model"].predict(test_mat)
        return preds

    @staticmethod
    def save_model(model, path):
        bst = model["model"]
        bst.save_model(path)

