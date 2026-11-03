import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger("feature_building_logger")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler) 

def feature_creation(processed_data_path, test_data_save_path, train_data_save_path):

    logger.info("Starting feature creation from %s", processed_data_path)

    df = pd.read_csv(processed_data_path, parse_dates= ["tpep_pickup_datetime"])
    logger.info("Loaded %d rows and %d columns", *df.shape)

    

    df["month"] = df["tpep_pickup_datetime"].dt.month

    df["day_of_the_week"] = df["tpep_pickup_datetime"].dt.day_of_week

    df.set_index("tpep_pickup_datetime", inplace= True)

    #creating lag features

    df = df.sort_values(["region", "tpep_pickup_datetime"])


    region_grp = df.groupby("region")

    lag_period = list(range(1,5))

    lag_features = region_grp["total_pickups_15min"].shift(lag_period)

    data = pd.concat([lag_features, df],axis = 1)
    rows_before_dropna = len(data)
    data.dropna(inplace = True)
    logger.info("Dropped %d rows with missing lag values, %d rows remain",
                rows_before_dropna - len(data), len(data))

    mapper = {name : f"lag_{ind+1}" for ind, name in enumerate(data.columns[0:4])}

    data= data.rename(columns= mapper)



    # split into test and train

    train_data = data.loc[data["month"].isin([1,2]), "lag_1": "day_of_the_week"]
    test_data = data.loc[data["month"].isin([3]), "lag_1": "day_of_the_week"]

    logger.info("Train rows (months 1-2): %d, test rows (month 3): %d",
                len(train_data), len(test_data))

    train_data.to_csv(train_data_save_path, index = True)
    logger.info("Saved train data to %s", train_data_save_path)

    test_data.to_csv(test_data_save_path, index = True)
    logger.info("Saved test data to %s", test_data_save_path)

if __name__ == "__main__":

    # resolve paths from this file's location so the script works from any cwd
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    processed_data_path = os.path.join(project_root, "data", "processed", "data_with_features.csv")

    train_data_save_path = os.path.join(project_root, "data", "processed", "train_data.csv")
    test_data_save_path = os.path.join(project_root, "data", "processed", "test_data.csv")

    os.makedirs(os.path.dirname(train_data_save_path), exist_ok=True)
    logger.info("Output directory ready: %s", os.path.dirname(train_data_save_path))

    feature_creation(processed_data_path, test_data_save_path, train_data_save_path)
    logger.info("Feature creation finished")








