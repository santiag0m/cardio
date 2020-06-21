from sklearn.ensemble import RandomForestClassifier

from .model import Model


class RandomForest(Model):
    def __init__(self, balanced=False):
        self.balanced = balanced

    def create_model(self, random_state):
        """
        Create a random forest model with tuned hyperparameters.

        Class balance is achived with biased random subsampling within
        decision trees.
        """
        if self.balanced:
            class_weight = "balanced"
        else:
            class_weight = None
        return RandomForestClassifier(
            class_weight=class_weight,
            # min_samples_leaf=0.008,
            n_estimators=25,
            # min_samples_split=75,
            max_depth=5,
            random_state=random_state,
        )

    def fit_model(self, model, train_data, train_labels):
        model.fit(train_data, train_labels)

    def predict_model(self, model, test_data):
        preds = model.predict_proba(test_data)
        preds = preds[:, 1]
        return preds
