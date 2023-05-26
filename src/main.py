import os
from typing import Dict, Any

import mlflow
# import optuna
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from .logger import logger
from .metrics import compute_metrics
from .utils import set_seed


def load_data():
    db = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(db.data, db.target)
    data = {}
    data["train"] = x_train
    data["train_labels"] = y_train
    data["test"] = x_test
    data["test_labels"] = y_test
    return data


def load_model(hparams: Dict[str, Any]):
    return RandomForestRegressor(
        n_estimators=hparams["n_estimators"],
        max_depth=hparams["max_depth"],
        max_features=3,
    )


def train_model(model, data, hparams):
    model = model.fit(data["train"], data["train_labels"])
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifacts(
        os.environ.get("MLFLOW_TRACKING_FOLDER"), artifact_path="rf_model.pkl"
    )
    return model


def predict(model, data):
    y_preds = model.predict(data["test"])
    compute_metrics(y_preds, data["test_labels"])

    return y_preds


def main():
    logger.info("Load dataset")
    data = load_data()

    logger.info("Set seed")
    seed = set_seed()

    with mlflow.start_run():
        hparams = {"n_estimators": 100, "max_depth": 6, "seed": seed}
        mlflow.log_params(hparams)

        logger.info("Load model")
        model = load_model(hparams)

        # logger.info("Start hyperparameter search")
        # hparams = hyperparameter_search(model, data)

        logger.info("Train model")
        model = train_model(model, data, hparams)

        logger.info("Predict")
        predictions = predict(model, data)


if __name__ == "__main__":
    main()
