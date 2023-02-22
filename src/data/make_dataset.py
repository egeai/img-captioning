# -*- coding: utf-8 -*-
import os
import json
import shutil
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import flow, task, get_run_logger
from zipfile import ZipFile

api = KaggleApi()
api.authenticate()


# @task(log_prints=True)
def unzip_data(raw_data_path: str, raw_zipped_file_path: str):
    logger = get_run_logger()
    logger.info("Unzipping the dataset file!")
    print(raw_zipped_file_path)
    if os.path.exists(raw_zipped_file_path):
        with ZipFile(raw_zipped_file_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_path)
    else:
        logger.info("Zip dataset file not exist!")


def export_metadata(metadata_list: list[dict[str, str]], path: str) -> None:
    """
    This function will prepare metadata for training data.
    :param metadata_list: metadata which has the list of file_name, text, vocab
    :param path: path for saving preprocessed data
    :return: None
    """
    # add metadata.jsonl file to this folder
    with open(path + "metadata.jsonl", 'w') as f:
        for item in metadata_list:
            f.write(json.dumps(item) + "\n")


# @task(log_prints=True)
def preprocess_data(raw_data_path: str,
                    keywords_path: str,
                    processed_data_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Read train csv file.")
    dataframe = pd.read_csv(raw_data_path)
    dataframe_keywords = pd.read_csv(keywords_path,
                                     on_bad_lines='skip',
                                     sep="\t\t",
                                     names=["id", "keywords"])
    dataframe = pd.merge(dataframe, dataframe_keywords, on="id")
    logger.info("Filter train data with 'chest x-ray' keywords")
    chest_x_ray_df = dataframe[
        dataframe.caption.str.contains("chest x-ray", case=False, na=False)
    ]
    chest_x_ray_df = chest_x_ray_df.drop("id", axis=1)
    logger.info("Rename train data columns to match the structure of HuggingFace.")
    chest_x_ray_df.rename(columns={'name': 'file_name', 'caption': 'text', 'keywords': 'vocab'}, inplace=True)
    logger.info("Prepare a metadata list from chest x-ray dataframe.")
    metadata_list = []
    for index, row in chest_x_ray_df.iterrows():
        metadata_list.append(
            {"file_name": row["file_name"], "text": row["text"], "vocab": row["vocab"].replace("\t", " ")})
    logger.info("Export metadata list to the processed folder as metadata.jsonl file.")
    export_metadata(metadata_list, processed_data_path)
    return chest_x_ray_df


# @task(log_prints=True)
def move_images_to_processed(chest_x_ray_data: pd.DataFrame, source_path: str, destination_path: str) -> None:
    """
    This function moves images from raw folder to processed folder
    :param chest_x_ray_data: Filtered DataFrame
    :param source_path: Source path of raw images folder
    :param destination_path: Destination path of raw images folder
    :return: None
    """
    logger = get_run_logger()
    for index, row in chest_x_ray_data.iterrows():
        colab_link = source_path + row['file_name']
        gdrive_link = destination_path
        shutil.copy(colab_link, gdrive_link)


# @task(log_prints=True)
def check_raw_data(raw_data_path: str) -> bool:
    logger = get_run_logger()
    if os.path.exists(raw_data_path + "all_data") \
            and os.path.exists(raw_data_path + "all_data/test") \
            and os.path.exists(raw_data_path + "all_data/train") \
            and os.path.exists(raw_data_path + "all_data/validation"):
        logger.info("All data folders are downloaded!")
    else:
        return False # raise Exception("Data folders aren't downloaded!")
    return True


# @task(log_prints=True, retries=3)
def fetch_data(raw_data_path: str, raw_zipped_file_path: str):
    logger = get_run_logger()
    try:
        if not os.path.exists(raw_zipped_file_path):
            logger.info("Fetching the dataset as zip file!")
            api.dataset_download_files("virajbagal/roco-dataset", path=raw_data_path, unzip=True)
        else:
            logger.info("File exist!")
    except (FileExistsError, FileNotFoundError) as error:
        raise Exception("Data folders aren't downloaded!", error)


@flow(name="Ingest flow")
def main_flow(cfg=None): # cfg type is DictConfig
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = get_run_logger()
    print(cfg.paths.raw.data)
    ret_val = check_raw_data(cfg.paths.raw.data)

    logger.info("Preprocess Train Data!")

    if ret_val is False:
        fetch_data(cfg.paths.raw.data, cfg.paths.raw.zipped_file)
    ret_val = check_raw_data(cfg.paths.raw.data)
    print(ret_val)
    """ 
    if ret_val is True:
        logger.info("Preprocess Train Data!")
        processed_train_df = preprocess_data(
            cfg.train.kaggle_radiology_data_path,
            cfg.train.kaggle_radiology_keywords_path,
            cfg.train.processed_data_path
        )
        move_images_to_processed(processed_train_df, cfg.train.images_raw_data_path, cfg.train.images_processed_data_path)

        logger.info("Preprocess Validation Data!")
        processed_val_df = preprocess_data(
            cfg.val.kaggle_radiology_data_path,
            cfg.val.kaggle_radiology_keywords_path,
            cfg.val.processed_data_path
        )
        move_images_to_processed(processed_val_df, cfg.val.images_raw_data_path, cfg.val.images_processed_data_path)

        logger.info("Preprocess Test Data!")
        processed_test_df = preprocess_data(
            cfg.test.kaggle_radiology_data_path,
            cfg.test.kaggle_radiology_keywords_path,
            cfg.test.processed_data_path
        )
        move_images_to_processed(processed_test_df, cfg.test.images_raw_data_path, cfg.test.images_processed_data_path)
        """
