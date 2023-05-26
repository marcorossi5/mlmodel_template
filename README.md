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
