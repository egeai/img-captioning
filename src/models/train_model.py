from tqdm import tqdm
from omegaconf import DictConfig
import gc


class Train:

    def __init__(self, model, train_loader, optimizer, epoch, scheduler, device, cfg: DictConfig):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.epoch = epoch
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg

    @staticmethod
    def learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def run(self) -> float:
        self.model.train()

        dataset_size = 0
        running_loss = 0.0
        epoch_loss = 0.0
        batch_size = 0.0

        number_of_batches = len(self.train_loader)

        bar = tqdm(enumerate(self.train_loader), total=number_of_batches)

        for step, data in bar:
            input_ids = data.pop('input_ids').to(self.device)
            attention_mask = data.pop('attention_mask').to(self.device)
            pixel_values = data.pop('pixel_values').to(self.device)

            if input_ids is not None:
                batch_size = input_ids.size(0)

            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 pixel_values=pixel_values,
                                 labels=input_ids)

            loss = outputs.loss
            # loss = loss / self.cfg.params.n_accumulate
            loss.backward()

            if (step + 1) % self.cfg.params.n_accumulate == 0:
                self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step(loss)

            running_loss += (loss.item() * batch_size)
            print("running_loss", running_loss)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                epoch=self.epoch,
                train_loss=epoch_loss,
                lr=self.optimizer.param_groups[0]['lr']
            )

        gc.collect()

        return epoch_loss


    """
    number_of_batches = len(self.train_loader)
    
    tqdm_obj = tqdm(self.train_loader, total=number_of_batches)
    epoch_loss = 0.0
    for i, batch in enumerate(tqdm_obj):
        outputs = self.model(
            input_ids=batch['input_ids'].squeeze().to(self.device),
            attention_mask=batch['attention_mask'].squeeze().to(self.device),
            pixel_values=batch['pixel_values'].squeeze().to(self.device))
            # return_loss=True)
        print(outputs)
        # loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        tqdm_obj.set_postfix(
            batch="{}/{}".format(i + 1, number_of_batches),
            train_loss=loss.item(),
            lr=self.learning_rate(self.optimizer)
        )
    epoch_loss = epoch_loss / number_of_batches
    return epoch_loss
    """
