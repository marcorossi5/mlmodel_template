# Template to train ML model

ML model template with MLFlow and Optuna

## Requirements

- [MLFlow](https://mlflow.org/)
- [Optuna](https://optuna.org)
- [u8darts](https://unit8co.github.io/darts/)

## Run the training with MLFlow

The services can be run through the following command:

```bash
./start.sh
```

Connect to `localhost:5000` in your favourite browser to check the MLFlow logs.

Alternatively, one can load the MLFlow logs alone running the utility script:

```bash
./mlflow_up.sh
```

And then shutting the service down with `./mlflow_down.sh`.

## Run without docker

First set up a python virtual environment installing the requirements:

```bash
pip install -r requirements.txt
```

In the `src/.env` file, moidify the `mlflow` environment variables to point to
local directories, e.g. `/path/to/folder/mlflow`.

Run the pipeline with the command:

```bash
python -m src.main
```

Visualize the recorded results in the UI with:

```bash
mlflow ui --backend-store-uri /path/to/folder/mlflow
```

## MLFlow

MLFlow is an experiment tracking framework ([docs](https://mlflow.org/docs/latest/tracking.html)).  
In this study repo, we are running MLFlow locally, that means that all the saved
data and artifacts are stored on the local disk.  
Many different other configurations are available and explained at
[this documentation page](https://mlflow.org/docs/latest/tracking.html).

The main idea behind the MLFlow implementation is that we can organize our trials
by experiments, which, in turn, are divided by runs.

Different runs can be compared directly from the browser through MLFlow UI, launched
by `mlflow ui` command.

Under each run, we can save many quantities, that persist in the MLFlow database
and can be retrieved at need:

- **parameters**: usually include models hyperparameters;
- **metrics**: scalar metrics to evaluate the model;
- **tags**: useful to add descriptions to the run.  
Particularly useful tag are system tags (available [here](https://mlflow.org/docs/latest/tracking.html#system-tags)).
For example, the `mlflow.note.content` accepts a string that will be rendered in
markdown to verbosely describe the content of the current run.
- **artifacts**: whatever file you want to upload and store in the MLFlow database.  
It will be visible and downloadable from the UI, or it can be inspected in a
script with the help of the [`MlflowClient`](https://mlflow.org/docs/latest/python_api/mlflow.client.html) class.

## Optuna

Link to [Optuna](https://optuna.org/): a framework for hyperparameter searching.

It can be used in synergy with MLFlow with the [`MLflowCallback`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html) object as follows:

```python
import optuna
import mlflow


mlflow_uri = "PATH/TO/MLFLOW/URI"
mlflc = MLflowCallback(
    tracking_uri=mlflow_uri,
    metric_name="study_metric",
    create_experiment=True,
)


@mlflc.track_in_mlflow()
def objective():
    """The function that computes the objective metric to minimize"""
    ...
    return metric

study = optuna.create_study(direction="minimize", study_name=experiment_name)
study.optimize(func, n_trials=n_trials, callbacks=[mlflc])
```

Within the `objective` function an MLFlow run is automatically created and quantities
like metrics, parameters, artifacts and tags can be logged manually calling one
of the following commands:

```python

import mlflow


@mlflc.track_in_mlflow()
def objective:
    ...
    mlflow.log_parameter("parameter_name": parameter)
    mlflow.log_metric("metric_name": metric)
    mlflow.log_artifact("PATH/TO/ARTIFACT")
    mlflow.log_tag("some_tag": tag)
```
