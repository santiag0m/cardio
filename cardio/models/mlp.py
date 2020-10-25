import os
import random
import shutil
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

import cardio
from .model import Model


TEMP_FOLDER = os.path.dirname(cardio.__file__)


class MLP(Model):
    def __init__(self, balanced=False, scaler="standard"):
        self.balanced = balanced
        if scaler == "minmax":
            self.scaler = preprocessing.MinMaxScaler()
        elif scaler == "standard":
            self.scaler = preprocessing.StandardScaler()
        else:
            raise ValueError(f"Scaler '{scaler}' is not supported.")

    def create_model(self, random_state):
        """
        Create an MLP (single hidden layer ANN) model with tuned hyperparameters.

        Class balance is implemented with weighted binary cross entropy.
        """
        return MLPClassifier(
            balanced=self.balanced,
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=256,
            learning_rate_init=0.01,
            max_iter=300,
            shuffle=True,
            random_state=random_state,
            momentum=0.9,
            validation_fraction=0.05,
            tol=-0.01,
            early_stopping=False,
            verbose=0,
        )

    def fit_model(self, model, train_data, train_labels):
        train_scaled = self.scaler.fit_transform(train_data)
        model.fit(train_scaled, train_labels)

    def predict_model(self, model, test_data):
        test_scaled = self.scaler.transform(test_data)
        preds = model.predict_proba(test_scaled)
        preds = preds[:, 0]
        return preds

    @staticmethod
    def save_model(model, path):
        model.model.save(path)


class MLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=1,
        learning_rate_init=0.1,
        max_iter=300,
        shuffle=True,
        random_state=None,
        momentum=0.9,
        validation_fraction=0.05,
        balanced=False,
        verbose=1,
        **kwargs,
    ):
        # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
        if random_state is not None:
            seed_value = random_state.get_state()[1][0]
        else:
            seed_value = 0
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

        if solver == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_init, beta_1=momentum
            )
        elif solver == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate_init, momentum=momentum
            )
        else:
            raise ValueError(f"The selected solver is not valid. (Got {solver})")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.regularization = alpha

        self.verbose = verbose
        self.shuffle = shuffle
        self.balanced = balanced
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.validation_fraction = validation_fraction

    def fit(self, train_data, train_label):
        assert len(train_data.shape) == 2, "'train_data' should be a 2D array"
        num_features = train_data.shape[-1]
        train_set, val_set = self._random_split(
            train_data=train_data, train_label=train_label,
        )
        if self.balanced:
            inverse_weight = (1 - train_label).sum() / train_label.sum()
        else:
            inverse_weight = 1

        self.model = KerasMLP(
            input_shape=num_features,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            regularization=self.regularization,
        )

        self.model.compile(
            optimizer=self.optimizer, loss=WeightedBce(inverse_weight), metrics=["acc"]
        )
        with tempfile.TemporaryDirectory() as temp_folder:
            filepath = os.path.join(temp_folder, "keras_best_nn")
            os.makedirs(filepath)

            self.model.fit(
                x=train_set[0],
                y=train_set[1],
                validation_data=val_set,
                epochs=self.max_iter,
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=filepath,
                        monitor="val_loss",
                        save_best_only=True,
                        save_weights_only=True,
                    )
                ],
                verbose=self.verbose,
            )
            self.model.load_weights(filepath)

    def predict_proba(self, test_data):
        test_data = test_data.astype(np.float32)
        return self.model.predict(test_data)

    def _random_split(self, train_data, train_label):
        val_index = int(self.validation_fraction * len(train_data))
        indexes = list(range(len(train_data)))
        random.shuffle(indexes)

        train_data = train_data[indexes, :]
        train_data = train_data.astype(np.float32)
        train_label = train_label[indexes]
        train_label = train_label.astype(np.float32)

        fit_data = train_data[val_index:, :]
        fit_label = train_label[val_index:]
        fit_label = np.expand_dims(fit_label, axis=-1)

        val_data = train_data[:val_index, :]
        val_label = train_label[:val_index]
        val_label = np.expand_dims(val_label, axis=-1)

        return (fit_data, fit_label), (val_data, val_label)


def KerasMLP(
    input_shape, hidden_layer_sizes=(15,), activation="relu", regularization=0.01
):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    regularizer = tf.keras.regularizers.l2(l=regularization)

    mlp = tf.keras.models.Sequential()
    for units in hidden_layer_sizes:
        mlp.add(
            tf.keras.layers.Dense(
                units, activation=activation, kernel_regularizer=regularizer
            )
        )
        mlp.add(tf.keras.layers.LayerNormalization())
    mlp.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    output = mlp(input_layer)

    return tf.keras.Model(inputs=input_layer, outputs=output)


def WeightedBce(class_weight):
    def loss(y_true, y_pred):
        weights = (y_true * (class_weight - 1)) + 1
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weighted_bce = tf.reduce_mean(bce * weights)
        return weighted_bce

    return loss
