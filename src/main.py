import logging
import os
from typing import Dict, Any

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from .logger import logger
from .metrics import compute_metrics
from .utils import set_seed


mlflc = MLflowCallback(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        metric_name="my metric score",
        create_experiment=True
    )


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
    mlflow.log_artifact("rf_model.pkl")
    return model


def objective(trial, data):
    max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    logger.info("Set seed")
    seed = set_seed()

    hparams = {"n_estimators": 100, "max_depth": max_depth, "seed": seed}
    mlflow.log_params(hparams)

    logger.info("Load model")
    model = load_model(hparams)

    logger.info("Train model")
    model = train_model(model, data, hparams)

    logger.info("Predict")
    y_preds = model.predict(data["test"])
    metrics = compute_metrics(y_preds, data["test_labels"])
    return metrics["rmse"]


def get_experiment_from_name(mlflow_client, experiment_name):
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        # experiment does not exist, create it
        logger.info("Create brand new experiment named: %s", experiment_name)
        experiment_id = mlflow_client.create_experiment(experiment_name)
        return mlflow_client.get_experiment(experiment_id)
    return experiment


def main():
    logger.info("Load dataset")
    data = load_data()
    
    experiment_name = os.environ.get("EXPERIMENT_NAME", "example_1")
    
    @mlflc.track_in_mlflow()
    def func(trial):
        return objective(trial, data)

    n_trials = 20
    logger.info("Hyperparameter search for %d trials", n_trials)
    
    original_level = logger.getEffectiveLevel()
    logger.setLevel(logging.WARNING)
    study = optuna.create_study(direction='minimize', study_name=experiment_name)

    study.optimize(func, n_trials=n_trials, callbacks=[mlflc])
    logger.setLevel(original_level)


if __name__ == "__main__":
    main()
