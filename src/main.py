#!/usr/bin/env python3

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from src.data.make_dataset import main_flow


def make_dir(path_of_folder: str, name: str) -> bool:
    is_exist = os.path.exists(path_of_folder)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path_of_folder)
        return True


#@task(log_prints=True)
#def



@hydra.main(version_base="1.3.1", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(cfg.paths.raw.data)
    main_flow(cfg)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[1]

    # mkdir data folder
    make_dir("data")

    # mkdir processed sub folder
    make_dir("data/processed")

    # mkdir raw sub folder
    make_dir("data/raw")

    # call the main function
    main()
