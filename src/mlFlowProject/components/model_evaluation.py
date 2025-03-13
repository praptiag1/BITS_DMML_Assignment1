from pathlib import Path
import os
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlFlowProject.entity.config_entity import ModelEvaluationConfig
from mlFlowProject.utils.common import save_json, get_features_from_store  # Add this import


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall, f1

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Fetch features from the feature store
        feature_refs = ["feature1", "feature2", "feature3"]  # Replace with actual feature names
        entity_df = pd.DataFrame({"entity_id": test_x.index})  # Replace with actual entity identifier
        features = get_features_from_store(entity_df, feature_refs)
        test_x = pd.concat([test_x, features], axis=1)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predictions = model.predict(test_x)

            (accuracy, precision, recall, f1) = self.eval_metrics(test_y, predictions)
            
            # Saving metrics as local
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1-score", f1)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="LightGBM")
            else:
                mlflow.sklearn.log_model(model, "model")