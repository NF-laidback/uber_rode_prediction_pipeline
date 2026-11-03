import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import logging
from yaml import safe_load
import os

logger = logging.getLogger("feature_extraction_logger")
logger.setLevel("DEBUG")




stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

def read_params(params_path = "params.yaml"):
    with open(params_path, "r") as f:
        params = safe_load(f)
    return params

data_input_path = "data/interim"

def data_loader(file_path, chunksize = 100000):
    df = pd.read_csv(file_path, chunksize= chunksize, usecols= ["pickup_latitude", "pickup_longitude"])

    return df

def save_model(model,model_path):
    """ function to save the trained models"""

    joblib.dump(model,model_path)

def save_processed_data(data_frame,save_data_path):
    data_frame.to_csv(save_data_path, index = False)

# read data to predict regions

def extract_features_pipeline(data_path, scaler_model, k_means_model, alpha, chunksize = 100000):

    logger.info(f"Starting feature extraction from {data_path} with chunksize={chunksize}")

    # count pickups per region in 15 min intervals, one chunk at a time to save memory
    counts = []

    reader = pd.read_csv(
        data_path,
        chunksize = chunksize,
        usecols = ["tpep_pickup_datetime", "pickup_latitude", "pickup_longitude"],
        parse_dates = ["tpep_pickup_datetime"],
    )

    for chunk_number, chunk in enumerate(reader, start = 1):
        logger.debug(f"Processing chunk {chunk_number} with {len(chunk)} rows")

        # assign each pickup to a region
        chunk["region"] = k_means_model.predict(
            scaler_model.transform(chunk[["pickup_longitude", "pickup_latitude"]])
        )

        partial_counts = (
            chunk.set_index("tpep_pickup_datetime")
            .groupby("region")["pickup_latitude"]
            .resample("15min")
            .count()
        )
        counts.append(partial_counts)

    # add up the counts from all chunks, since the same region and time bin can appear in several chunks
    total_counts = pd.concat(counts).groupby(level = ["region", "tpep_pickup_datetime"]).sum()
    logger.info(f"Counted pickups across {len(counts)} chunks into {len(total_counts)} region/interval bins")

    # fill empty 15 min intervals with 0 so every region has a continuous timeline
    regions = []

    for region, series in total_counts.groupby(level = "region"):
        series = series.droplevel("region")
        full_index = pd.date_range(series.index.min(), series.index.max(), freq = "15min")
        series = series.reindex(full_index, fill_value = 0)

        regions.append(pd.DataFrame({
            "region": region,
            "tpep_pickup_datetime": full_index,
            "total_pickups_15min": series.values,
        }))

    resampled_data = pd.concat(regions, ignore_index = True)
    logger.info(f"Filled timeline for {len(regions)} regions, {len(resampled_data)} rows in total")

    # replacing zero pickups with a min value of 10
    epsilon = 10
    resampled_data["total_pickups_15min"] = resampled_data["total_pickups_15min"].replace(0, epsilon)

    # calculating average pickups for each region in 15 min time intervals using EWMA
    # shift is applied inside each region so the first interval of a region does not use another region's data
    resampled_data["average_pickup"] = (
        resampled_data.groupby("region")["total_pickups_15min"]
        .transform(lambda s: s.ewm(alpha = alpha).mean().shift(1))
        .round()
    )

    logger.info("Finished feature extraction")
    return resampled_data



if __name__ == "__main__":
    data_path = os.path.join(data_input_path + "/data_without_outliers.csv")
    logger.info(f"Input data: {data_path}")

    chunks = data_loader(data_path)

    scaler = StandardScaler()

    logger.info("Fitting scaler")
    for chunk in chunks:
        scaler.partial_fit(chunk)

    # save the scaler

    model_path = "models/scaler.joblib"
    save_model(scaler, model_path)
    logger.info(f"Saved scaler to {model_path}")

    chunks = data_loader(data_path)
    params = read_params()["extract_features"]["mini_batch_kmeans"]
    alpha = read_params()["extract_features"]["ewma"]["alpha"]
    logger.info(f"Loaded params: mini_batch_kmeans={params}, ewma alpha={alpha}")

    mini_batch_kmeans = MiniBatchKMeans(**params)

    logger.info("Fitting MiniBatchKMeans")
    for chunk in chunks:
        scaled_chunk = scaler.transform(chunk)

        mini_batch_kmeans.partial_fit(scaled_chunk)

    # save the model

    k_means_model_path = "models/k_means.joblib"
    save_model(mini_batch_kmeans, k_means_model_path)
    logger.info(f"Saved KMeans model to {k_means_model_path}")

    resampled_df = extract_features_pipeline(data_path, scaler, mini_batch_kmeans, alpha)

    data_save_path = "data/processed/data_with_features.csv"

    save_processed_data(resampled_df, data_save_path)
    logger.info(f"Saved {len(resampled_df)} feature rows to {data_save_path}")


    









