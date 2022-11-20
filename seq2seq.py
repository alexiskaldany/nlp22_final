""" 
seq2seq.py
A sequence-to-sequence model for text summarization.
Data: https://www.kaggle.com/datasets/timmayer/covid-news-articles-2020-2022
author: @alexiskaldany
created on 11/14/2022

The first model we built was a classification model. Having discovered this is reasonably easy to build, this model will predict the headline based on the content of the article.

It also used a pre-trained model from Huggingface. For this model I will try to build a model using primarily PyTorch.

Content -> Headline

"""
# Kaggle Api (Download Data from Kaggle, Run only 1 time)
# import os
# os.environ['KAGGLE_USERNAME'] = 'koyanjo'
# os.environ['KAGGLE_KEY'] = '33bfba07e0815efc297a1a4488dbe6a3'
# import kaggle
# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()
# api.dataset_download_files('timmayer/covid-news-articles-2020-2022')
# import zipfile
# with zipfile.ZipFile('covid-news-articles-2020-2022.zip', 'r') as zipref:
#     zipref.extractall()
# csv_path = os.getcwd() + "/covid_articles_raw.csv"

""" 
seq2seq.py
A sequence-to-sequence model for text summarization.
Data: https://www.kaggle.com/datasets/timmayer/covid-news-articles-2020-2022
author: @alexiskaldany
created on 11/14/2022

The first model we built was a classification model. Having discovered this is reasonably easy to build, this model will predict the headline based on the content of the article.

It also used a pre-trained model from Huggingface. For this model I will try to build a model using another transformer.

content -> title



"""
import time 
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import gc
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig,get_scheduler
from transformers.trainer_seq2seq import Seq2SeqTrainer
import os
from torchmetrics import Perplexity

## Configs
metrics = Perplexity()
max_len = int ( 2048 )
lr = float ( 1e-5 )
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
config = BartConfig.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", config=config)
optimizer = AdamW(model.parameters(), lr=lr)
class summarizationDataLoader(Dataset):
    """
    CustomDataLoader object
    """

    def __init__(self, dataframe: pd.DataFrame , tokenizer, max_len: int = 2048):
        self.content = dataframe["content"].tolist()
        self.title = dataframe["title"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        inputs = self.tokenizer(
            self.content[index],
            max_length=self.max_len,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs["labels"] = self.tokenizer(
            self.title[index],
            max_length=self.max_len,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )['input_ids']
        return inputs
    
### Loading the data
save_path = os.getcwd() + "/seq2seq_results.csv"
print(f"Save path: {save_path}")
# csv_path = os.getcwd() + "/covid_articles_raw.csv"
csv_path = os.getcwd() + "/covid_articles_raw_first_250_.csv"
print(f"Loading data from {csv_path}")
dataset = pd.read_csv(csv_path, encoding="utf-8")[:500]
print(f"Loaded {len(dataset)} rows of data")

train_dataset = dataset.sample(frac=0.8, random_state=0)
print(f"Loaded {len(train_dataset)} rows of training data")
val_dataset = dataset.drop(train_dataset.index).sample(frac=0.5, random_state=0)
print(f"Loaded {len(val_dataset)} rows of validation data")
test_dataset = dataset.drop(train_dataset.index).drop(val_dataset.index)
print(f"Loaded {len(test_dataset)} rows of test data")

train_dataloader = summarizationDataLoader(train_dataset, tokenizer, max_len)
print(train_dataloader[0])
val_dataloader = summarizationDataLoader(val_dataset, tokenizer, max_len)
test_dataloader = summarizationDataLoader(test_dataset, tokenizer, max_len)

if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
if torch.has_mps:
    device = torch.device("mps")
if not torch.cuda.is_available():
    device = torch.device("cpu")
    
print(f"Using device: {device}")

class trainingBartSummarizer:
    def __init__(self,
        model,
        max_len,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device, 
        num_epochs=1,
        lr=5e-5,
        optimizer=None
    ):
        self.model = model
        self.max_len = max_len
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
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
        self.device = device
        self.metric_list = []
        self.pbar = tqdm(total=self.num_steps_total)
        
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
        metric_dict["correct"] = (
            self.train_dataloader.category[index]
            if mode == "train"
            else self.val_dataloader.category[index]
        )
        metric_dict["predicted"] = self.config.id2label[
            torch.argmax(outputs.logits, dim=1).item()
        ]
        metric_dict["true_positive"] = (
            1 if metric_dict["correct_label"] == metric_dict["predicted_label"] else 0
        )
        metric_dict["total_epoch_loss"] = total_epoch_loss
        metric_dict["elapsed_time"] = elapsed_time
        
        self.metric_list.append(metric_dict)
    
    # def create_epoch_statistics(self, current_epoch: int) -> None:
    #     epoch_statistics = pd.DataFrame(
    #         columns=["epoch", "mode", "accuracy", "loss", "elapsed_time"]
    #     )
    #     df = pd.DataFrame(self.metric_list)
    #     epoch_dfs = [df[df["epoch"] == epoch] for epoch in df["epoch"].unique()]
    #     for index, epoch in enumerate(epoch_dfs):
    #         train_df = epoch[epoch["mode"] == "train"]
    #         train_df["cumsum_tp"] = train_df["true_positive"].cumsum()
    #         train_df["accuracy"] = train_df["cumsum_tp"] / train_df.index
    #         epoch_statistics.loc[index] = [
    #             current_epoch,
    #             "train",
    #             train_df["accuracy"].iloc[-1],
    #             train_df["loss"].iloc[-1],
    #             train_df["elapsed_time"].iloc[-1],
    #         ]

    #         val_df = epoch[epoch["mode"] == "val"]
    #         val_df["cumsum_tp"] = val_df["true_positive"].cumsum()
    #         val_df["accuracy"] = val_df["cumsum_tp"] / val_df.index
    #         epoch_statistics.loc[index + 1] = [
    #             current_epoch,
    #             "val",
    #             val_df["accuracy"].iloc[-1],
    #             val_df["loss"].iloc[-1],
    #             val_df["elapsed_time"].iloc[-1],
    #         ]
    #     self.epoch_statistics = epoch_statistics
    #     self.epoch_statistics.to_csv(os.getcwd() + "/epoch_statistics.csv", index=False)

    def saving_stats(self, save_path: str) -> pd.DataFrame:
        headers = [
            "mode",
            "epoch",
            "index",
            "loss",
            "correct",
            "predicted",
            "perplexity",
            "elapsed_time",
        ]
        df = pd.DataFrame(self.metric_list, columns=headers)
        df.to_csv(save_path)
        print(f"Saved metrics to {save_path}")
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
        self.mode = "train"
        self.epoch = 0
        self.accuracy = 0
        self.pbar = tqdm(total=self.num_steps_total)
        self.model.resize_token_embeddings(len(train_dataloader.tokenizer))
        self.model.to(device)
        train_start = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        self.step_count = 0
        ### Train
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.mode = "train" 
            total_train_loss = 0
            self.model.train()
            for index, batch in enumerate(self.train_dataloader):
                print(type(batch))
                self.step_count += 1
                self.model.zero_grad()
                outputs = self.model(**batch.to(device))
                loss = outputs[0]
                # total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                current_time = time.time()
                elapsed_time = current_time - train_start
                self.tracking_outputs(
                    outputs, epoch, index, "train", total_train_loss, elapsed_time)
                self.pbar.update(1)
                self.pbar.set_postfix({"acc": self.accuracy})
            ### Validate
            self.model.eval()
            total_eval_loss = 0
            self.mode = "val"
            # Evaluate data for one epoch
            for index, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch.to(device))
                    loss = outputs[0]
                    total_eval_loss += loss.item()
                elapsed_time = current_time - train_start
                self.step_count += 1
                self.tracking_outputs(
                    outputs, epoch, index, "val", total_eval_loss, elapsed_time
                )
                self.pbar.update(1)
                self.pbar.set_postfix({"acc": self.accuracy})
           
            # if save_weights:
            #     self.model.save_pretrained(model_weights_dir)
            #     self.tokenizer.save_pretrained(model_weights_dir)
            #     logger.info(f"Saved model weights to {model_weights_dir}")
            # self.create_epoch_statistics()
        
bart_summarizer = trainingBartSummarizer(model = model, max_len=max_len,train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader,device=device,num_epochs=1,lr=lr,optimizer=optimizer)

bart_summarizer.train()
bart_summarizer.saving_stats(save_path=os.getcwd() + "/metrics.csv")