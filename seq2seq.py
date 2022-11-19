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
import os
os.environ['KAGGLE_USERNAME'] = 'koyanjo'
os.environ['KAGGLE_KEY'] = '33bfba07e0815efc297a1a4488dbe6a3'
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('timmayer/covid-news-articles-2020-2022')
import zipfile
with zipfile.ZipFile('covid-news-articles-2020-2022.zip', 'r') as zipref:
    zipref.extractall()
csv_path = os.getcwd() + "/covid_articles_raw.csv"