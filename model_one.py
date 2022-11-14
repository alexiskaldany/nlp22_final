import gc
import os
import sys
import time
import pandas as pd
import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
    get_scheduler,
)

from typing import List, Tuple, Dict, Any, Optional
logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="INFO",
    backtrace=True,
    colorize=True,
)

"""  
TODO:
1. Build better statistics for output
2. Determine tradeoff  between size and accuracy (taking a long time to train)
3. Possibly use a fast tokenizer or some smaller transfomer
4. Save model weights
5. Create testing method

"""

class CustomDataLoader(Dataset):
    """
    CustomDataLoader object
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            tokenizer,
            max_len: int = 2048,
            label2id: dict = None,
    ):
        """
        Params:
            dataframe (pd.DataFrame): whichever slice of the data
            tokenizer (_type_): tokenizer object
            max_len (int, optional): Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.content = dataframe["content"].tolist()
        self.category = self.data["category"].tolist()
        self.max_len = max_len
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.metric_list = []

    def __len__(self):
        """
        Params:
            self: instance of object
        Returns:
            number of samples
        """
        return len(self.content)

    def __getitem__(self, index):
        inputs = self.tokenizer(
            self.content[index],
            max_length=self.max_len,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs["labels"] = torch.nn.functional.one_hot(
            torch.tensor([self.label2id[self.category[index]]]), num_classes=5
        ).to(torch.float)
        return inputs


class TrainingDistilBert:
    """
    Object to hold model, tokenizer, and dataloaders
    """

    def __init__(
            self,
            config,
            model,
            max_len,
            train_dataloader,
            val_dataloader,
            test_dataloader,
    ):
        """
        config: DistilBertConfig object
        model: DistilBertForSequenceClassification object
        max_len: int
        train_dataloader: CustomDataLoader object
        val_dataloader: CustomDataLoader object
        test_dataloader: CustomDataLoader object
        """
        self.config = config
        self.model = model
        self.max_len = max_len
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.metric_list = []
        logger.info("Initialized DistilBertForSequenceClassification object")

    def set_train_parameters(self, num_epochs=1, lr=5e-5, optimizer=None):
        """
        num_epochs (int, optional): Defaults to 1.
        lr (float, optional): Defaults to 5e-5.
        optimizer (torch.optim, optional): Defaults to None.
        """
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.num_validation_steps = self.num_epochs * len(self.val_dataloader)
        self.num_steps_total = self.num_epochs * (
                len(self.train_dataloader) + len(self.val_dataloader)
        )
        self.lr = lr
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        logger.info("Set training parameters")

    def tracking_outputs(
            self,
            outputs,
            epoch,
            index,
            mode,
            total_epoch_loss,
            elapsed_time,
    ):
        metric_dict = {}
        metric_dict["mode"] = mode
        metric_dict["epoch"] = epoch
        metric_dict["index"] = index
        metric_dict["loss"] = outputs[0].item()
        metric_dict["correct_label"] = (
            self.train_dataloader.category[index]
            if mode == "train"
            else self.val_dataloader.category[index]
        )
        metric_dict["predicted_label"] = self.config.id2label[
            torch.argmax(outputs.logits, dim=1).item()
        ]
        metric_dict["true_positive"] = (
            1 if metric_dict["correct_label"] == metric_dict["predicted_label"] else 0
        )
        metric_dict["total_epoch_loss"] = total_epoch_loss
        metric_dict["elapsed_time"] = elapsed_time
        metric_dict["step_count"] = self.step_count
        self.metric_list.append(metric_dict)

    def update_to_terminal(
            self, current_epoch: int, current_index: int, mode: str, elapsed_time: int
    ) -> None:
        logger.debug(f"self.metric_list: {self.metric_list}")
        if self.metric_list == []:
            logger.debug("self.metric_list is empty")
            return
        headers = [
            "mode",
            "epoch",
            "index",
            "loss",
            "correct_label",
            "predicted_label",
            "true_positive",
            "total_epoch_loss",
            "elapsed_time",
        ]
        updated_to_terminal = pd.DataFrame(self.metric_list, columns=headers)
        length_of_epoch = (
            self.train_dataloader.__len__()
            if mode == "train"
            else self.val_dataloader.__len__()
        )
        whole_epoch_length = (
                self.train_dataloader.__len__() + self.val_dataloader.__len__()
        )
        number_of_steps_run = (current_epoch) * whole_epoch_length + current_index
        remaining_steps = self.num_steps_total - number_of_steps_run
        average_time_per_step = number_of_steps_run / elapsed_time
        remaining_time_seconds = remaining_steps * average_time_per_step
        minutes, seconds = divmod(remaining_time_seconds, 60)
        minutes = round(int(minutes), 0)
        seconds = round(int(seconds), 0)
        number_of_correct_predictions = updated_to_terminal[updated_to_terminal['epoch'] == current_epoch][
            "true_positive"].sum()
        if number_of_correct_predictions == 0 or current_index == 0:
            accuracy = 0
        else:
            accuracy = number_of_correct_predictions / current_index
        logger.info(
            f"{self.step_count} | {minutes}:{seconds} remaining | E: {current_epoch} | {mode.capitalize()} | {round(self.step_count / self.num_steps_total, 2)} | T: {number_of_correct_predictions} | F: {current_index - number_of_correct_predictions} | Accuracy: {round(accuracy, 2)}")

    def create_epoch_statistics(self, current_epoch: int) -> None:
        epoch_statistics = pd.DataFrame(columns=['epoch', 'mode', 'accuracy', 'loss', 'elapsed_time'])
        df = pd.DataFrame(self.metric_list)
        epoch_dfs = [df[df["epoch"] == epoch] for epoch in df["epoch"].unique()]
        for index, epoch in enumerate(epoch_dfs):
            train_df = epoch[epoch["mode"] == "train"]
            train_df['cumsum_tp'] = train_df['true_positive'].cumsum()
            train_df['accuracy'] = train_df['cumsum_tp'] / train_df.index
            epoch_statistics.loc[index] = [current_epoch, "train", train_df['accuracy'].iloc[-1],
                                           train_df['loss'].iloc[-1], train_df['elapsed_time'].iloc[-1]]

            val_df = epoch[epoch["mode"] == "val"]
            val_df['cumsum_tp'] = val_df['true_positive'].cumsum()
            val_df['accuracy'] = val_df['cumsum_tp'] / val_df.index
            epoch_statistics.loc[index + 1] = [current_epoch, "val", val_df['accuracy'].iloc[-1],
                                               val_df['loss'].iloc[-1], val_df['elapsed_time'].iloc[-1]]
        self.epoch_statistics = epoch_statistics
        self.epoch_statistics.to_csv(os.getcwd() + "/epoch_statistics.csv", index=False)

    def saving_stats(self, save_path: str) -> pd.DataFrame:
        headers = [
            "mode",
            "epoch",
            "index",
            "loss",
            "correct_label",
            "predicted_label",
            "true_positive",
            "total_epoch_loss",
            "elapsed_time",
        ]
        df = pd.DataFrame(self.metric_list, columns=headers)
        df.to_csv(save_path)
        logger.info(f"Saved metrics to {save_path}")
        return df

    def train(
            self,
            eval_during_training: bool = False,
            save_weights: bool = True,
            model_weights_dir="./results/model_weights/",
    ):
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

        self.model.resize_token_embeddings(len(train_dataloader.tokenizer))
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        logger.info(f"Using device..{device}")
        self.model.to(device)
        train_start = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        self.step_count = 0
        ### Train
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            logger.info("Training...")
            total_train_loss = 0
            self.model.train()
            for index, batch in enumerate(self.train_dataloader):
                self.step_count += 1
                [batch[key][0].to(device) for key in batch.keys()]
                self.model.zero_grad()
                outputs = self.model(**batch)
                loss = outputs[0]
                total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                current_time = time.time()
                elapsed_time = current_time - train_start
                self.update_to_terminal(epoch, index, "train", elapsed_time)
                self.tracking_outputs(
                    outputs, epoch, index, "train", total_train_loss, elapsed_time
                )
            logger.info(f"Finished training epoch {epoch + 1}")
            logger.info("Validating...")
            ### Validate
            self.model.eval()
            total_eval_loss = 0
            # Evaluate data for one epoch
            for index, batch in enumerate(self.val_dataloader):
                [batch[key][0].to(device) for key in batch.keys()]
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs[0]
                    logits = outputs[1]
                    total_eval_loss += loss.item()
                elapsed_time = current_time - train_start
                self.step_count += 1
                self.update_to_terminal(epoch, index, "val", elapsed_time)
                self.tracking_outputs(
                    outputs, epoch, index, "val", total_eval_loss, elapsed_time
                )

            logger.info(f"Finished validating epoch {epoch + 1}")
            # if save_weights:
            #     self.model.save_pretrained(model_weights_dir)
            #     self.tokenizer.save_pretrained(model_weights_dir)
            #     logger.info(f"Saved model weights to {model_weights_dir}")
            # self.create_epoch_statistics()
        logger.info("Finished training")


### Configs
max_len = 2048
label2id = {"business": 0, "general": 1, "tech": 2, "science": 3, "esg": 4}
id2label = {0: "business", 1: "general", 2: "tech", 3: "science", 4: "esg"}
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", model_max_length=max_len
)
configuration = DistilBertConfig(
    num_labels=5,
    id2label=id2label,
    label2id=label2id,
    max_position_embeddings=max_len,
)
# update_index = 5
### Loading the data
save_path = os.getcwd() + "/testerino_epochs_stats.csv"
logger.info(f" Save path: {save_path}")
# csv_path = os.getcwd() + "/covid_articles_raw.csv"
csv_path = os.getcwd() + '/covid_articles_raw_first_250_.csv'
logger.info(f"Loading data from {csv_path}")
dataset = pd.read_csv(csv_path, encoding="utf-8")[:100]
logger.info(f"Loaded {len(dataset)} rows of data")

train_dataset = dataset.sample(frac=0.8, random_state=0)
logger.info(f"Loaded {len(train_dataset)} rows of training data")
val_dataset = dataset.drop(train_dataset.index).sample(frac=0.5, random_state=0)
logger.info(f"Loaded {len(val_dataset)} rows of validation data")
test_dataset = dataset.drop(train_dataset.index).drop(val_dataset.index)
logger.info(f"Loaded {len(test_dataset)} rows of test data")

train_dataloader = CustomDataLoader(train_dataset, tokenizer, max_len, label2id)
val_dataloader = CustomDataLoader(val_dataset, tokenizer, max_len, label2id)
test_dataloader = CustomDataLoader(test_dataset, tokenizer, max_len, label2id)

### Training
model = DistilBertForSequenceClassification(config=configuration)
DistilBert_Model = TrainingDistilBert(
    config=configuration,
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    max_len=max_len,
)
DistilBert_Model.set_train_parameters(num_epochs=5)
DistilBert_Model.train(eval_during_training=True)

### Saving the stats

DistilBert_Model.saving_stats(save_path)
