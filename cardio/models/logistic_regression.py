from sklearn import preprocessing, linear_model

from .model import Model


class LogisticRegression(Model):
    def __init__(self, balanced=False, scaler="minmax"):
        self.balanced = balanced
        if scaler == "minmax":
            self.scaler = preprocessing.MinMaxScaler()
        elif scaler == "standard":
            self.scaler = preprocessing.StandardScaler()
        else:
            raise ValueError(f"Scaler '{scaler}' is not supported.")

    def create_model(self, random_state):
        """
        Create a logistic regression model with tuned hyperparameters.

        Class balance is implemented with inverse weight coefficients.
        """
        if self.balanced:
            class_weight = "balanced"
        else:
            class_weight = None
        return linear_model.LogisticRegression(
            class_weight=class_weight,
            max_iter=200,
            fit_intercept=False,
            random_state=random_state,
        )

    def fit_model(self, model, train_data, train_labels):
        train_scaled = self.scaler.fit_transform(train_data)
        model.fit(train_scaled, train_labels)

    def predict_model(self, model, test_data):
        test_scaled = self.scaler.transform(test_data)
        preds = model.predict_proba(test_scaled)
        preds = preds[:, 1]
        return preds
