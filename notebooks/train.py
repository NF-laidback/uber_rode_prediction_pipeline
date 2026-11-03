import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config
from pathlib import Path

import joblib


from sklearn.linear_model import LinearRegression



import logging

logger = logging.getLogger("train_model_logger")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)



if not logger.handlers:
    logger.addHandler(handler)


current_path = Path(__file__)
root = current_path.parent.parent

def train_models(data_path):

    logger.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates= ["tpep_pickup_datetime"])
    logger.info("Loaded %d rows and %d columns", *df.shape)

    df.set_index("tpep_pickup_datetime", inplace= True)

    X_train = df.drop(columns = ["total_pickups_15min"])
    y_train = df["total_pickups_15min"]

    encoder = ColumnTransformer(
        [("cat", OneHotEncoder(drop = "first", sparse_output= False), ["region", "day_of_the_week"])],
        remainder= "passthrough", n_jobs = -1
    )

    logger.info("Fitting encoder on %d features", X_train.shape[1])
    encoder.fit(X_train)

    X_train_encoded = encoder.transform(X_train)
    logger.debug("Encoded feature matrix shape: %s", X_train_encoded.shape)

    lr = LinearRegression()

    logger.info("Fitting LinearRegression")
    lr.fit(X_train_encoded, y_train)
    logger.info("Training R^2: %.4f", lr.score(X_train_encoded, y_train))

    return encoder, lr


def save_model(model,model_path):
    logger.info("Saving %s to %s", type(model).__name__, model_path)
    joblib.dump(model,model_path)

if __name__ == "__main__":
    logger.info("Starting training pipeline")
    set_config(transform_output= "pandas")

    data_path = root/"data/processed/train_data.csv"

    encoder_save_path = root/"models/encoder.joblib"

    model_save_path = root/"models/lr_model.joblib"

    encoder, lr = train_models(data_path)

    save_model(encoder, encoder_save_path)
    save_model(lr, model_save_path)

    logger.info("Training pipeline finished")

