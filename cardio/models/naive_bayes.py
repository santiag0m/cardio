import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from .model import Model
from cardio.utils import get_bootstrap_metrics, get_metrics


class NaiveBayes(Model):
    GAUSSIAN_VARIABLES = {
        "Paciente: Edad",  # Age
        "Indice de masa corporal",  # BMI
        "Fracción de Eyección (E)",  # LVEF
        "Ultimo hematocrito",  # Hematocrit
        "Ultimo nivel de creatinina",  # Creatinine
    }

    BERNOULLI_VARIABLES = {
        "Paciente: Sexo_Femenino",  # Gender: Female
        "Hipertensión",  # Hypertension
        "Diabetes",  #  Diabetes
        "Enfermedad pulmonar crónica",  # COPD
        "Enfermedad arterial periférica",  # Peripheral Artery Disease
        "Enfermedad cerebro vascular",  # Stroke
        "Insuficiencia cardíaca",  # Heart failure
        "Insuficiencia renal - diálisis",  # Dialysis
        "Endocarditis infecciosa",  # Endocarditis
        "Resucitación",  # Reanimation
        "Shock cardiogénico",  # Cardiogenic Shock
        "Arritmia",  # Arrhythmia
        "Peso del procedimiento - procedimiento aislado no CABG",  # (Weight of procedure) Isolated non-CABG
        "Peso del procedimiento - dos procedimientos",  # (Weight of procedure) Two procedures
        "Peso del procedimiento - tres o más procedimientos",  # (Weight of procedure) Three or more procedures
    }

    ORDINAL_VARIABLES = {
        "Número de vasos coronarios enfermos",  # Coronary Arteries Blocked
        "Insuficiencia aórtica (E)",  # Aortic valve insufficiency
        "Insuficiencia mitral (E)",  # Mitral valve regurgitation
        "Insuficiencia tricuspídea (E)",  # Tricuspid valve regurgitation
        "Estado",  # Urgency upon admission
    }

    def create_model(self, *args, **kwargs):
        """
        Create a naive bayes model.

        Class balance is not implemented given the nature of the model.
        """
        model = {
            "gaussian": GaussianNB(),
            "bernoulli": BernoulliNB(),
        }

        return model

    def fit_model(self, model, train_data, train_labels):
        model["gaussian"].fit(train_data["gaussian"], train_labels)
        model["bernoulli"].fit(train_data["bernoulli"], train_labels)

    def predict_model(self, model, test_data):
        gaussian_preds = model["gaussian"].predict_proba(test_data["gaussian"])[:, 1]
        bernoulli_preds = model["bernoulli"].predict_proba(test_data["bernoulli"])[:, 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            preds = np.log(gaussian_preds) + np.log(bernoulli_preds)
        preds = np.exp(preds)

        return preds

    def run_experiment(
        self, train, test, baseline=None, bootstrap=False, *args, **kwargs
    ):
        assert train["features"] == test["features"]
        train = self._format_variable_data(train)
        test = self._format_variable_data(test)

        if bootstrap:
            metric_fun = get_bootstrap_metrics
        else:
            metric_fun = get_metrics

        for _ in tqdm(range(1)):  # Output Consistency
            model = self.create_model()
            self.fit_model(
                model,
                train_data={
                    "gaussian": train["gaussian_data"],
                    "bernoulli": train["bernoulli_data"],
                },
                train_labels=train["labels"],
            )
            pred_test = self.predict_model(
                model,
                test_data={
                    "gaussian": test["gaussian_data"],
                    "bernoulli": test["bernoulli_data"],
                },
            )
        metrics = metric_fun(test["labels"], pred_test, baseline=baseline)
        metrics["outputs"] = pred_test
        return metrics

    def _format_variable_data(self, split):
        split = {**split}
        split["gaussian_data"] = self._get_variables(self.GAUSSIAN_VARIABLES, split)
        split["bernoulli_data"] = self._get_variables(self.BERNOULLI_VARIABLES, split)

        ordinal_df = self._get_variables_df(self.ORDINAL_VARIABLES, split)
        temp_array = None
        for var_name in ordinal_df.columns:
            tempdf = pd.get_dummies(ordinal_df[var_name])
            if temp_array is None:
                temp_array = tempdf.values
            else:
                temp_array = np.concatenate([temp_array, tempdf.values], axis=1)

        split["bernoulli_data"] = np.concatenate(
            [split["bernoulli_data"], temp_array], axis=1
        )
        return split

    @staticmethod
    def _get_variables(variable_set, split):
        idxs = []
        for var_name in variable_set:
            try:
                idx = split["features"].index(var_name)
                idxs.append(idx)
            except ValueError:
                continue
        return split["data"][:, idxs]

    @staticmethod
    def _get_variables_df(variable_set, split):
        df = {}
        for var_name in variable_set:
            try:
                idx = split["features"].index(var_name)
                df[var_name] = split["data"][:, idx]
            except ValueError:
                continue
        return pd.DataFrame(df)
