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