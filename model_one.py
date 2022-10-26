from transformers import (
    DistilBertModel,
    DistilBertConfig,
    TrainingArguments,
    Trainer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
    get_scheduler,
)
from loguru import logger
import torch
import gc
from torch.optim import AdamW
from torch.utils.data import Dataset
import pandas as pd

# from tqdm.auto import tqdm
import time
import sys

logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="INFO",
    backtrace=True,
    colorize=True,
)


label2id = {"business": 0, "general": 1, "tech": 2, "science": 3, "esg": 4}
id2label = {0: "business", 1: "general", 2: "tech", 3: "science", 4: "esg"}
#############
# configuration = DistilBertConfig(
#     num_labels=5,
#     id2label=id2label,
#     label2id=label2id,
#     max_position_embeddings=2048,
# )

# model = DistilBertForSequenceClassification(config=configuration)

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased",model_max_length=2048)

# data_250 = pd.read_csv("covid_articles_raw_first_250_.csv", encoding="utf-8")

# first_content = data_250["content"][0]
# first_content_tokens = tokenizer(first_content, return_tensors="pt", padding=True, truncation=True,max_length=2048)

# first_category = data_250["category"][0]
# first_category_id = configuration.label2id[first_category]
# labels = torch.nn.functional.one_hot(
#     torch.tensor([first_category_id]), num_classes=5
# ).to(torch.float)
# first_content_tokens["labels"] = labels

# logits = model(**first_content_tokens).logits

# # logits = model(**first_content_tokens,labels=labels).logits
# prediction = model.config.id2label[logits.argmax().item()]
# print(prediction)

##################
# loss = model(**first_content_tokens, labels=labels)[0]
# print(loss)

# logits = model(**first_content_tokens, labels=labels)
# prediction = torch.argmax(logits.logits, dim=1)
# print(prediction)


# print(configuration.id2label)
# configuration.label2id = {0: 'business', 1: 'general', 2: 'tech', 3: 'science', 4: 'esg'}
# print(configuration.label2id)

# dataset = pd.read_csv('covid_articles_train.csv', encoding='utf-8')

# train_dataset = dataset.sample(frac=0.8, random_state=0)
# val_dataset = dataset.drop(train_dataset.index).sample(frac=0.5, random_state=0)
# test_dataset = dataset.drop(train_dataset.index).drop(val_dataset.index)


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
        self.content = dataframe["content"]
        self.category = self.data["category"]
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


class DistilBertForSequenceClassification:
    """
    Object to hold model, tokenizer, and dataloaders
    """

    def __init__(
        self, model, config, max_len, train_dataloader, val_dataloader, test_dataloader ,update_index)-> None:
        self.model = model
        self.config = config
        self.max_len = max_len
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.update_index: update_index
        self.metric_list = []
        logger.info("Initialized DistilBertForSequenceClassification object")

    def set_train_parameters(self, num_epochs=1, lr=5e-5, optimizer=None):
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.num_validation_steps = self.num_epochs * len(self.val_dataloader)
        self.num_steps_total = self.num_epochs * (len(self.train_dataloader) + len(self.val_dataloader))
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
        epoch: int,
        index: int,
        mode: str,
        total_train_loss: float,
        elapsed_time: float,
    ):
        metric_dict = {}
        metric_dict["mode"] = mode
        metric_dict["epoch"] = epoch
        metric_dict["index"] = index
        metric_dict["loss"] = outputs[0]
        metric_dict["correct_label"] = self.category[index]
        metric_dict["predicted_label"] = self.config.id2label[
            torch.argmax(outputs.logits, dim=1).item()
        ]
        metric_dict["true_positive"] = (
            1 if metric_dict["correct_label"] == metric_dict["predicted_label"] else 0
        )
        metric_dict["total_train_loss"] = total_train_loss
        metric_dict["elapsed_time"] = elapsed_time
        self.metric_list.append(metric_dict)

    def update_to_terminal(self,current_epoch: int, current_index: int, mode: str:) -> None:
        df = pd.DataFrame(self.metric_list)
        current_epoch_df = df[df["epoch"] == current_epoch]
        length_of_epoch = len(train_dataloader) if mode == "train" else len(val_dataloader)
        whole_epoch_length = len(train_dataloader) + len(val_dataloader)
        number_of_steps_run = (current_epoch - 1)*whole_epoch_length + current_index
        remaining_steps = self.num_steps_total - number_of_steps_run
        average_time_per_step = number_of_steps_run / int(df["elapsed_time"])
        remaining_time_seconds = remaining_steps * average_time_per_step
        logger.info(f"Estimated time remaining: {remaining_time_seconds/60} minutes")
        number_of_correct_predictions = current_epoch_df["true_positive"].sum()
        accuracy = number_of_correct_predictions / length_of_epoch
        logger.info(f"Epoch: {current_epoch} | {mode} | {current_index*100}/{length_of_epoch} | Accuracy: {accuracy}")
    
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

        self.model.resize_token_embeddings(len(self.tokenizer))
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        logger.info("Using device..", device)

        self.model.to(device)
        train_start = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            logger.info("Training...")
            total_train_loss = 0
            self.model.train()
            for index, batch in enumerate(self.train_dataloader):
                [batch[key][0].to(device) for key in batch.keys()]
                self.model.zero_grad()
                outputs = self.model(**batch)
                loss = outputs[0]
                total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                if index % update_index == 0:
                    self.update_to_terminal(epoch, index, "train")
    
                current_time = time.time()
                epoch_time = current_time - epoch_start
                elapsed_time = current_time - train_start
                self.tracking_outputs(
                    outputs, epoch, index, "train", total_train_loss, elapsed_time
                )


# print(datacollator)

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)


# train = pd.read_csv('covid_articles_train.csv', encoding='utf-8')


# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_imdb["train"],
#     eval_dataset=tokenized_imdb["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )
