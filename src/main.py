import logging
import os

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

from .logger import logger
from .utils import set_seed
from .pipeline import load_data, PipelineSklLinRegressor


mlflc = MLflowCallback(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        metric_name="study_metric",
        create_experiment=True,
)


def objective(trial, data):
    logger.info("Set seed")
    seed = set_seed()
    mlflow.log_param("seed", seed)

    pipe = PipelineSklLinRegressor()

    logger.info("Suggest hyperparameters")
    hparams = pipe.suggest_hparams(trial)
    
    logger.info("Load model")
    pipe.load_model(hparams)

    logger.info("Train model")
    pipe.train_model(data, hparams)

    logger.info("Predict")
    metrics = pipe.test_model(data)

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
    
    experiment_name = os.environ.get("EXPERIMENT_NAME", "default_experiment")
    
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
