# Template to train ML model

ML model template with MLFlow and Optuna

## Requirements

- [MLFlow](https://mlflow.org/)
- [Optuna](https://optuna.org)
- torch

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
