#!/usr/bin/env python3

import os
import time
import gc

from IPython.core.display_functions import display
from PIL import Image
from pathlib import Path
import hydra
from datasets import load_dataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from transformers import (CLIPModel,
                          CLIPProcessor,
                          AutoModelForCausalLM,
                          AutoProcessor,
                          BlipForConditionalGeneration,
                          VisionEncoderDecoderModel
                          )

from models.train_model import Train
from data.image_captioning_dataset import ImageCaptioningDataset
from src.data.make_dataset import data_structure_flow
from src.models.valid_model import Validation

logging.set_verbosity_error()
torch.cuda.empty_cache()


def make_dir(path_of_folder: str) -> bool:
    is_exist = os.path.exists(path_of_folder)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path_of_folder)
        return True


def test_and_predict(model, processor, device, cfg: DictConfig) -> list:
    predictions = []
    model.eval()
    test_dataset = load_dataset("imagefolder", data_dir=cfg.params.data_dir, split="test")
    test_ds = ImageCaptioningDataset(dataset=test_dataset, processor=processor)

    test_dataloader = torch.utils.data.DataLoader(test_ds, collate_fn=collate_fn,
                                                  batch_size=cfg.params.train_batch_size)
    for batch in test_dataloader:
        # 'input_ids' 'attention_mask' 'pixel_values'
        input_ids = batch.pop('input_ids').to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        pixel_values = batch.pop('pixel_values').to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=input_ids)
        logits = outputs.logits
        yhat = torch.argmax(logits, dim=-1)
        predictions.append(yhat)
    return predictions


def plot_history(history):
    train_history = history['train']
    dev_history = history['dev']
    plt.plot(list(range(1, len(train_history) + 1)), train_history, label="train loss")
    plt.plot(list(range(1, len(dev_history) + 1)), dev_history, label="dev loss")
    plt.xticks(list(range(1, len(train_history) + 1)))
    plt.xlabel("epoch")
    plt.legend()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def training(model, train_dataloader, val_dataloader, device, cfg: DictConfig):
    model.to(device)
    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                        patience=cfg.params.patience,
                                                        factor=cfg.params.factor)
    best_val_score = float('inf')
    train_history = []
    val_history = []
    number_of_bad_epochs = 0

    print("Learning phase")
    print("Used device: ", device)
    print("------------------------------")

    for epoch in range(1, cfg.params.max_epochs + 1):
        print("Epoch {:03d}/{:03d}".format(epoch, cfg.params.max_epochs))

        if number_of_bad_epochs >= cfg.params.max_bad_epochs:
            print(
                "Epoch {:03d}/{:03d}: exiting training after too many bad epochs.".format(epoch, cfg.params.max_epochs))
            torch.save(model.state_dict(), "final.pt")
            break
        else:
            epoch_start_time = time.time()

            train = Train(model=model, train_loader=train_dataloader,
                          optimizer=optimizer,
                          epoch=cfg.params.epochs,
                          scheduler=lr_scheduler,
                          device=device,
                          cfg=cfg)
            epoch_train_loss = train.run()

            validate = Validation(model=model, val_dataloader=val_dataloader,
                                  epoch=epoch,
                                  max_epochs=cfg.params.max_epochs,
                                  device=device)
            epoch_val_score = validate.run()

            duration = time.time() - epoch_start_time

            lr_scheduler.step(epoch_val_score)
            train_history.append(epoch_train_loss)

            val_history.append(epoch_val_score)

            if epoch_val_score < best_val_score:
                number_of_bad_epochs = 0
                best_val_score = epoch_val_score
                torch.save(model.state_dict(), "best.pt")
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - SAVED (NEW) BEST "
                      "MODEL. Duration: {:.3f} s".format(epoch,
                                                         cfg.params.max_epochs,
                                                         epoch_train_loss,
                                                         epoch_val_score,
                                                         duration))
            else:
                number_of_bad_epochs += 1
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - NUMBER OF BAD "
                      "EPOCH.S: {}. Duration: {:.3f} s".format(epoch,
                                                               cfg.params.max_epochs,
                                                               epoch_train_loss,
                                                               epoch_val_score,
                                                               number_of_bad_epochs,
                                                               duration))

    history = {'train': train_history, 'dev': val_history}
    return history


@hydra.main(version_base="1.3.1", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # set device to cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # read and prepare dataset structure
    # data_structure_flow(cfg)

    # build vocabulary (no-need-right-now)
    # initialize, load model and processor from pre-trained
    processor = AutoProcessor.from_pretrained(cfg.params.microsoft_pretrained)
    model = AutoModelForCausalLM.from_pretrained(cfg.params.microsoft_pretrained).to("cuda")
    # load train, valid and test datasets
    # root = cfg.params.data_folder # "./data/processed/"

    train_dataset = load_dataset("imagefolder", data_dir=cfg.params.data_dir, split="train")
    # val_dataset = load_dataset("imagefolder", data_dir=cfg.params.data_dir, split="validation")
    # test_dataset = load_dataset("imagefolder", data_dir=cfg.params.data_dir, split="test")

    # creating train and validate Dataset and DataLoader
    train_ds = ImageCaptioningDataset(dataset=train_dataset, processor=processor)
    """
    print(train_ds[0].keys())

    val_ds = ImageCaptioningDataset(dataset=val_dataset, processor=processor)

    train_dataloader = torch.utils.data.DataLoader(train_ds, collate_fn=collate_fn,
                                                   batch_size=cfg.params.train_batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_ds, collate_fn=collate_fn,
                                                 batch_size=cfg.params.valid_batch_size)

    history = training(model, train_dataloader, val_dataloader, device=device, cfg=cfg)
    # plot the history
    plot_history(history)
    """

    example = train_dataset[0]
    image = example["image"]
    width, height = image.size
    img = Image.open(image.resize((int(0.3 * width), int(0.3 * height))))
    display(img)

    # predictions = test_and_predict(model=model, processor=processor, device=device, cfg=cfg)
    # print(predictions[0])


def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[1]
    # mkdir data folder
    make_dir("data")
    # mkdir processed sub folder
    make_dir("data/processed")
    # train sub folders
    make_dir("data/processed/train")
    make_dir("data/processed/train/images")
    # validation sub folders
    make_dir("data/processed/validation")
    make_dir("data/processed/validation/images")
    # test sub folders
    make_dir("data/processed/test")
    make_dir("data/processed/test/images")
    # mkdir raw sub folder
    make_dir("data/raw")
    # call the main function

    # report_gpu()

    main()
