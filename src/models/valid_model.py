from tqdm import tqdm


class Validation:

    def __init__(self, model, val_dataloader, epoch, max_epochs, device):
        self.model = model
        self.val_dataloader = val_dataloader
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.device = device

    def run(self) -> float:
        self.model.eval()
        number_of_batches = len(self.val_dataloader)
        tqdm_obj = tqdm(self.val_dataloader, total=number_of_batches)
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm_obj):

            input_ids = batch.pop('input_ids').squeeze().to(self.device)
            attention_mask = batch.pop('attention_mask').squeeze().to(self.device)
            pixel_values = batch.pop('pixel_values').squeeze().to(self.device)

            outputs = self.model(input_ids=input_ids,
                                 # attention_mask=attention_mask,
                                 pixel_values=pixel_values,
                                 labels=input_ids)

            """
            outputs = self.model(
                input_ids=batch.pop('input_ids').squeeze(),
                attention_mask=batch.pop('attention_mask').squeeze(),
                pixel_values=batch.pop('pixel_values').squeeze())
                # return_loss=True)
            """

            # loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
            loss = outputs.loss
            epoch_loss += loss.item()
            tqdm_obj.set_postfix(
                batch="{}/{}".format(i+1, number_of_batches),
                dev_loss=loss.item()
            )
        epoch_loss = epoch_loss / number_of_batches
        return epoch_loss
