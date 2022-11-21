from transformers import pipeline
import pandas as pd

summary = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

df = pd.read_csv("/Users/alexiskaldany/school/nlp22_final/covid_articles_raw.csv")
summary_list = []

for i in range(len(df)):
    summary_list.append(summary(df["content"][i], max_length=150, min_length=30, do_sample=False))
    if i % 100 == 0:
        print(i)

df['summary'] = summary_list

df.to_csv("/Users/alexiskaldany/school/nlp22_final/covid_articles_summary.csv")
    