from typing import Dict, Any

import mlflow
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

