from transformers import (
    DistilBertModel,
    DistilBertConfig,
    TrainingArguments,
    Trainer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import OrderedDict

""" 
5 different values in ['category'] column:
"""
"""
business,264440
general,150700
tech,52258
science,8096
esg,2045
"""
label2id = {"business": 0, "general": 1, "tech": 2, "science": 3, "esg": 4}
id2label={0: "business", 1: "general", 2: "tech", 3: "science", 4: "esg"}
#############
configuration = DistilBertConfig(
    num_labels=5,
    id2label=id2label,
    label2id=label2id,
    max_position_embeddings=512,
)

model = DistilBertForSequenceClassification(config=configuration)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased",model_max_length=1024)

data_250 = pd.read_csv("covid_articles_raw_first_250_.csv", encoding="utf-8")

first_content = data_250["content"][0]
first_content_tokens = tokenizer(first_content, return_tensors="pt", padding=True, truncation=True,max_length=512)

first_category = data_250["category"][0]
first_category_id = configuration.label2id[first_category]
labels = torch.nn.functional.one_hot(
    torch.tensor([first_category_id]), num_classes=5
).to(torch.float)

logits = model(**first_content_tokens,labels=labels).logits
prediction = model.config.id2label[logits.argmax().item()]
print(prediction)

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

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int = 512,label2id: dict = None):
        """
        Params:
            dataframe (pd.DataFrame): whichever slice of the data
            tokenizer (_type_): tokenizer object
            max_len (int, optional): Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.content = dataframe["content"]
        self.target = self.data["category"]
        self.max_len = max_len
        self.label2id = label2id

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
        inputs["labels"] = torch.nn.functional.one_hot(torch.tensor([self.label2id[self.target[index]]]), num_classes=5).to(torch.float)
        return inputs




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
