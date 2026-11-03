import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob
import yaml

import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)["data_ingestion"]

data_folder_path = "data/raw"

import logging

logger = logging.getLogger("file_logger")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("log.log")
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def data_reader(file_path):

    file_pattern = os.path.join(data_folder_path, "*.csv")

    logger.info("Searching for CSV files with pattern: %s", file_pattern)
    files = sorted(glob.glob(file_pattern))

    if not files:
        logger.error("No CSV files found matching: %s", file_pattern)
        raise FileNotFoundError(f"No CSV files found matching: {file_pattern}")

    logger.info("Found %d CSV file(s) to read", len(files))

    columns_to_be_used = ['tpep_pickup_datetime', 'tpep_dropoff_datetime','trip_distance', 'pickup_longitude',
       'pickup_latitude','dropoff_longitude', 'dropoff_latitude','fare_amount']

    data_frame= dd.read_csv(files, usecols = columns_to_be_used, parse_dates = ["tpep_pickup_datetime"])
    logger.info("Lazily loaded %d partition(s) with columns: %s", data_frame.npartitions, columns_to_be_used)

    return  data_frame

# range for latitude and longitude

min_latitude = params["min_latitude"]
max_latitude = params["max_latitude"]
min_longitude = params["min_longitude"]
max_longitude = params["max_longitude"]

# inlier range for fare amount and trip distance
min_fare_amount_val = params["min_fare_amount"]
max_fare_amount_val = params["max_fare_amount"]
min_trip_distance_val = params["min_trip_distance"]
max_trip_distance_val = params["max_trip_distance"]


def dask_pipeline(data_frame):

    logger.info("Starting outlier filtering pipeline")

    data_frame_ = data_frame[(data_frame["trip_distance"].between(min_trip_distance_val,  max_trip_distance_val, inclusive = "both")) & (data_frame["fare_amount"].between(min_fare_amount_val, max_fare_amount_val, inclusive = "both"))& (data_frame["pickup_latitude"].between(min_latitude, max_latitude))
                &(data_frame["dropoff_latitude"].between(min_latitude, max_latitude)) & (data_frame["pickup_longitude"].between(min_longitude, max_longitude))
                &(data_frame["dropoff_longitude"].between(min_longitude, max_longitude))]


    cols_to_drop = ['tpep_dropoff_datetime','trip_distance','dropoff_longitude', 'dropoff_latitude','fare_amount']

    df = data_frame_.drop(cols_to_drop, axis = 1)

    logger.info("Computing filtered data (dropping columns: %s)", cols_to_drop)
    result = df.compute()
    logger.info("Outlier filtering complete: %d rows, %d columns remaining", len(result), result.shape[1])

    return result

def save_files(data_frame):
    save_path = "data/interim"

    output_file = save_path + "/data_without_outliers.csv"
    logger.info("Saving %d rows to %s", len(data_frame), output_file)
    data_frame.to_csv(output_file, index = False)
    logger.info("Save complete: %s", output_file)



if __name__ == "__main__":

    logger.info("Data ingestion pipeline started")

    data_frame = data_reader(data_folder_path)

    df_final = dask_pipeline(data_frame)

    save_files(df_final)

    logger.info("Data ingestion pipeline finished")







