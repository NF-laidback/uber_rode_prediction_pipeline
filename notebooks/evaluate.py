import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error

import dagshub
import logging

import json

import joblib
from pathlib import Path
import mlflow

dagshub.init(repo_owner= "NF-laidback", repo_name= "Uber_rode_prediction_pipeline", mlflow= True)

mlflow.set_tracking_uri("https://dagshub.com/NF-laidback/uber_rode_prediction_pipeline.mlflow")

mlflow.set_experiment("DVC Pipeline")
set_config(transform_output= "pandas")

# create a logger
logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

current_file_path = Path(__file__)
root = current_file_path.parent.parent

encoder_path = root/"models/encoder.joblib"

lr_model_path = root/"models/lr_model.joblib"

data_path = root/"data/processed/test_data.csv"

training_data_path = root/"data/processed/train_data.csv"

json_path = root/"run_info.json"

def save_run_info(run_id, artifact_path, model_uri_path, json_path):
    run_info = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri_path": model_uri_path
    }

    with open(json_path, "w") as f:
        json.dump(run_info, f)


if __name__ == "__main__":
    logger.info("Loading test data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates= ["tpep_pickup_datetime"])

    df.set_index("tpep_pickup_datetime", inplace= True)

    X_test = df.drop(columns= ["total_pickups_15min"])
    y_test = df["total_pickups_15min"]
    logger.info("Test data loaded: %d rows, %d feature columns", len(df), X_test.shape[1])

    logger.info("Loading encoder from %s and model from %s", encoder_path, lr_model_path)
    encoder = joblib.load(encoder_path)

    lr_model = joblib.load(lr_model_path)

    X_test_encoded = encoder.transform(X_test)
    logger.info("Encoded test features: shape %s", X_test_encoded.shape)

    y_pred = lr_model.predict(X_test_encoded)
    logger.info("Predictions generated for %d rows", len(y_pred))

    loss = mean_absolute_percentage_error(y_test,y_pred)
    logger.info("Test MAPE: %.4f", loss)


    with mlflow.start_run(run_name= "lr_model"):
        logger.info("Started MLflow run 'lr_model'")

        mlflow.log_params(lr_model.get_params())

        mlflow.log_metric("MAPE", loss)

        training_data = mlflow.data.from_pandas(pd.read_csv(training_data_path, parse_dates= ["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
                                                ,targets = "total_pickups_15min")

        test_data = mlflow.data.from_pandas(pd.read_csv(data_path, parse_dates= ["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
                                                ,targets = "total_pickups_15min")


        mlflow.log_input(training_data, "training")
        mlflow.log_input(test_data, "test")
        logger.info("Logged training and test datasets to MLflow")

        model_signature = mlflow.models.infer_signature(X_test_encoded, y_pred)

        logged_model = mlflow.sklearn.log_model(lr_model,"demand prediction",signature= model_signature,
                                                pip_requirements="requirements.txt")

        run_id = logged_model.run_id
        artifact_path = logged_model.artifact_path
        model_uri = logged_model.model_uri
        logger.info("Model logged: run_id=%s, model_uri=%s", run_id, model_uri)

        save_run_info(run_id, artifact_path,model_uri, json_path)
        logger.info("Run info saved to %s", json_path)






