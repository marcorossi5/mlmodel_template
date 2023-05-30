import glob
import os
from typing import Dict, Any

from darts import TimeSeries, models
from darts.metrics import metrics as darts_metrics
import mlflow
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from . import metrics

def load_data():
    db = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(db.data, db.target)
    data = {}
    data["train"] = x_train
    data["train_labels"] = y_train
    data["test"] = x_test
    data["test_labels"] = y_test
    return data


def load_m5_data():
    data_folder = os.environ.get("M5_FOLDER")
    glob_str = os.path.join(data_folder, "m5_*.pkl")
    file_list = sorted(glob.glob(glob_str))
    series_list = [TimeSeries.from_pickle(f) for f in file_list]
    data = {}
    data["train"] = [s[:-28].astype(np.float32) for s in series_list]
    data["test"] = [s[-28:].astype(np.float32) for s in series_list]
    return data


class Pipeline:
    def suggest_hparams():
        """Suggest hyperparameters"""
    
    def load_model():
        """Load the model"""
        
    def train_model():
        """Train the model"""
    
    def test_model():
        """Test the model"""


class PipelineSklLinRegressor(Pipeline):
    def suggest_hparams(self, trial):
        max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        return {"max_depth": max_depth}
    
    def load_model(self, hparams: Dict[str, Any]):
        mlflow.log_param("model", "sklearn_linear_regressor")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=hparams["max_depth"],
            max_features=3,
        )
    
    def train_model(self, data, hparams):
        model = self.model.fit(data["train"], data["train_labels"])
        with open("rf_model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact("rf_model.pkl")
    
    def test_model(self, data):
        y_preds = self.model.predict(data["test"])
        
        # compute metrics
        mae = metrics.mae(y_preds, data["test_labels"])
        rmse = metrics.rmse(y_preds, data["test_labels"])

        metrics_dict = {"rmse": rmse, "mae": mae}
        mlflow.log_metrics(metrics_dict)
        return metrics_dict


class PipelineNHiTS(Pipeline):
    def suggest_hparams(self, trial):
        lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32,64,128,256,512])
        return {"lr": lr, "batch_size": batch_size}
    
    def load_model(self, hparams: Dict[str, Any]):
        mlflow.log_param("model", "nhits")
        self.model : models.NHiTSModel = models.NHiTSModel(
            input_chunk_length=28*4,
            output_chunk_length=28,
            n_epochs=1
        )
    
    def train_model(self, data, hparams):
        model = self.model.fit(data["train"])
        model.save("nhits_model.pkl")
        mlflow.log_artifact("nhits_model.pkl")
    
    def test_model(self, data):
        y_preds = self.model.predict(28, series=data["train"])
        
        # compute metrics
        mae = darts_metrics.mae(y_preds, data["test"])
        rmse = darts_metrics.rmse(y_preds, data["test"])
        mase = darts_metrics.mase(y_preds, data["test"], insample=data["train"])

        metrics_dict = {
            "rmse": np.mean(rmse),
            "rmse_med": np.median(rmse),
            "rmse_std": np.std(rmse),
            "rmse_min": np.min(rmse),
            "rmse_max": np.max(rmse),
            "mae": np.mean(mae),
            "mae_med": np.median(mae),
            "mae_std": np.std(mae),
            "mae_min": np.min(mae),
            "mae_max": np.max(mae),
            "mase": np.mean(mase),
            "mase_med": np.median(mase),
            "mase_std": np.std(mase),
            "mase_min": np.min(mase),
            "mase_max": np.max(mase),
        }

        mlflow.log_metrics(metrics_dict)
        return metrics_dict

