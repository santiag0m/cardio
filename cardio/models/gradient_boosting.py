import numpy as np
import xgboost as xgb

from .model import Model


class GradientBoosting(Model):
    def __init__(self, balanced=False):
        self.balanced = balanced
        self.feature_importance = None

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
        self.feature_importance = bst.get_score(importance_type="gain")

    def predict_model(self, model, test_data):
        test_mat = xgb.DMatrix(data=test_data)
        preds = model["model"].predict(test_mat)
        return preds

